'''
arm_sdk_publisher.py

Sends arm joint targets to the G1 over the rt/arm_sdk topic, while the
robot's own walking controller (g1_ctrl) keeps balancing the legs.
Used by SimController in mimic mode.

Dry run (publishes nothing) unless constructed with real=True.
Not tested on the physical robot yet.
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import math
import threading
import time

import numpy as np

from joint_smoothing import JointSmoother
from g1_gains import ARM_SDK_KD, ARM_SDK_KP
from g1_joint_limits import G1_29DOF_JOINT_LIMITS


#----------------------------------------------------------
# Constants
#----------------------------------------------------------
# {motor index: (lower limit, upper limit)} in rad, used to clamp every target
_LIMITS = {lim.index: (lim.lower, lim.upper) for lim in G1_29DOF_JOINT_LIMITS.values()}

ARM_SDK_JOINTS = list(range(12, 29))    # waist (12-14) + arms (15-28)
ARM_ONLY_JOINTS = list(range(15, 29))   # arms only (default)
WEIGHT_INDEX = 29                       # motor_cmd[29].q carries the blend weight


#----------------------------------------------------------
# Target validation
#----------------------------------------------------------
def validate_targets(targets, allowed):
    '''{index: angle} -> cleaned dict, or ValueError. Clamps to the real limits.'''
    cleaned = {}
    for index, value in targets.items():
        index = int(index)
        # refuse any motor we are not allowed to move (e.g. the legs)
        if index not in allowed:
            raise ValueError(f"Motor {index} is not an allowed target")
        value = float(value)
        # refuse nan / inf before they reach the robot
        if not math.isfinite(value):
            raise ValueError(f"Motor {index} received a non-finite target")
        # clamp into the joint's physical range
        lower, upper = _LIMITS[index]
        cleaned[index] = min(max(value, lower), upper)
    return cleaned


#----------------------------------------------------------
# Publisher
#----------------------------------------------------------
class ArmSdkPublisher:
    def __init__(self, interface=None, domain_id=0, real=False, control_hz=50.0,
                 ramp_s=2.0, max_command_speed=3.0, smoothing_tau=0.08, log=print, joints=ARM_ONLY_JOINTS):
        self.joints = list(joints)      # which motors we are allowed to drive
        self.real = bool(real)          # False = dry run, nothing is published
        if self.real and not interface:
            raise ValueError("real=True needs the network interface (e.g. 'lo' for the sim, enp3s0 for the robot)")
        self.interface = interface
        self.domain_id = domain_id
        self.dt = 1.0 / control_hz      # seconds between published messages
        self.ramp_s = ramp_s            # seconds to ramp the weight 0 <-> 1
        self.max_command_speed = max_command_speed   # rad/s limit on the arm targets
        self.smoothing_tau = smoothing_tau           # smoothing time constant (s)
        self.log = log

        # blend weight: current value and where it is heading
        self.weight = 0.0
        self.weight_target = 0.0

        # threading: one background thread publishes, the main thread sets targets
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()  # set once the starting pose is known
        self.thread = None

        self.shaper = None              # JointSmoother, created in _set_start()
        self.start_q = None             # arm pose at startup (what disengage() returns to)
        self.latest_q = None            # most recent measured joint positions
        self.sent = 0                   # messages published so far

    #------------------------------------------------------
    # Connection
    #------------------------------------------------------
    def init(self, timeout=10.0, start_q=None):
        '''Real: connect and wait for rt/lowstate. Dry-run: start from start_q
        (or zeros) with no DDS at all.'''
        if not self.real:
            q = np.zeros(29) if start_q is None else np.asarray(start_q, dtype=float)
            self._set_start(q)
            self.log("[arm_sdk] DRY RUN: nothing is published. Pass real=True to drive a robot.")
            return

        # unitree imports are inside init() so a dry run works without the SDK
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()    # message reused every tick
        ChannelFactoryInitialize(self.domain_id, self.interface)

        # publisher: arm commands out on rt/arm_sdk
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()

        # subscriber: joint positions in from rt/lowstate
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._on_low_state, 10)

        # block until the first lowstate arrives (gives us start_q)
        if not self.ready.wait(timeout):
            raise TimeoutError(f"No rt/lowstate from the robot on {self.interface} (domain {self.domain_id})")

    def _on_low_state(self, msg):
        '''Callback for every rt/lowstate message.'''
        q = np.array([msg.motor_state[i].q for i in range(29)])
        with self.lock:
            self.latest_q = q
        # the very first message defines the starting pose
        if not self.ready.is_set():
            self._set_start(q)

    def _set_start(self, q):
        '''Remember the starting pose and build the joint smoother from it,
        so the arms begin exactly where they are (no jump).'''
        self.start_q = q.copy()
        self.shaper = JointSmoother(q, self.dt, tau_s=self.smoothing_tau, max_speed=self.max_command_speed, engage_s=self.ramp_s)
        self.ready.set()

    #------------------------------------------------------
    # Engage / targets
    #------------------------------------------------------
    def start(self):
        '''Start the background publishing thread.'''
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="arm-sdk", daemon=True)
        self.thread.start()

    def engage(self):
        '''Ramp the arm_sdk weight to 1 (take the arms from locomotion).'''
        with self.lock:
            self.weight_target = 1.0

    def set_targets(self, targets):
        '''{motor index: angle in rad}. Validated and clamped, then handed to
        the smoother, which smooths and speed-limits them.'''
        cleaned = validate_targets(targets, set(self.joints))
        with self.lock:
            for i, v in cleaned.items():
                self.shaper.set_target(i, v)

    def disengage(self, timeout=None):
        '''Ease the arms back to their start pose, then ramp the weight to 0
        (locomotion takes the arms back). Blocks until done or timeout.'''
        # 1) send the arms back to where they started
        with self.lock:
            for i in self.joints:
                self.shaper.set_target(i, self.start_q[i])
        deadline = time.monotonic() + (timeout or (3.0 + 2 * self.ramp_s))

        # 2) wait until every arm joint is within 0.03 rad of its start pose
        while time.monotonic() < deadline:
            with self.lock:
                back = np.all(np.abs(self.shaper.q[self.joints] - self.start_q[self.joints]) < 0.03)
            if back:
                break
            time.sleep(0.02)

        # 3) hand the arms back to locomotion by ramping the weight to 0
        with self.lock:
            self.weight_target = 0.0
        while time.monotonic() < deadline and self.weight > 0.0:
            time.sleep(0.02)
        return self.weight == 0.0       # True if the handover finished in time

    def measured_positions(self):
        '''Latest rt/lowstate joint positions (29,), or None.'''
        with self.lock:
            return None if self.latest_q is None else self.latest_q.copy()

    def stop(self):
        '''Stop the publishing thread (does NOT disengage, call disengage() first).'''
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    #------------------------------------------------------
    # Publishing loop
    #------------------------------------------------------
    def step_weight(self):
        '''Move the weight one small step towards its target (never a jump).'''
        step = self.dt / self.ramp_s if self.ramp_s > 0 else 1.0
        self.weight = float(np.clip(self.weight + np.clip(self.weight_target - self.weight, -step, step), 0.0, 1.0))
        return self.weight

    def build_message(self):
        '''One tick -> (weight, {index: q}). No DDS, testable.'''
        with self.lock:
            w = self.step_weight()
            q = self.shaper.tick()
        return w, {i: float(q[i]) for i in self.joints}

    def _run(self):
        '''Background thread: build and publish one message every dt seconds.'''
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            w, q = self.build_message()
            if self.real:
                self._publish(w, q)
            # dry run: print one status line per second instead
            elif self.sent % int(1.0 / self.dt) == 0:
                self.log(f"[arm_sdk dry-run] weight {w:.2f} "
                         + " ".join(f"{i}:{v:+.2f}" for i, v in q.items()))
            self.sent += 1
            # fixed schedule (not sleep(dt)) so timing errors don't accumulate
            next_tick += self.dt
            self.stop_event.wait(max(0.0, next_tick - time.monotonic()))

    def _publish(self, weight, q):
        '''Fill in the LowCmd_ message and write it to rt/arm_sdk.'''
        cmd = self.low_cmd
        cmd.motor_cmd[WEIGHT_INDEX].q = weight      # blend weight rides in motor 29's q
        for i, value in q.items():
            if not math.isfinite(value):
                raise ValueError(f"non-finite arm_sdk target for motor {i}")
            m = cmd.motor_cmd[i]
            m.q = value         # target angle
            m.dq = 0.0          # no velocity feed-forward
            m.tau = 0.0         # no torque feed-forward
            m.kp = ARM_SDK_KP   # stiffness
            m.kd = ARM_SDK_KD   # damping
        cmd.crc = self.crc.Crc(cmd)     # the robot rejects messages with a bad CRC
        self.publisher.Write(cmd)