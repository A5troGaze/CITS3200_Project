"""CycloneDDS rt/lowcmd publisher that drives the Unitree MuJoCo G1 from
person_id joint targets. SIMULATOR ONLY: never run this against a real robot
(on the robot, rt/lowcmd bypasses balance; use arm_sdk_publisher.py).

* Targets arrive as {motor_index: angle_rad} at vision rate (~30 Hz) and are
  validated (finite, allowed joint, clamped to the real limits).
* A fixed-rate thread (default 500 Hz) shapes them (command_shaping.py:
  low-pass + velocity cap + engage ramp) and publishes LowCmd with the SDK
  example's KP/KD (g1_gains.py) and a gravity feed-forward `tau` for the
  commanded joints (g1_sim.GravityComp), so the SDK's gains do not let the
  arms and torso sag.
* Joints that are not commanded:
    hold mode (default)     held at their first observed position with the
                            same gains, as before. The legs are then stiff
                            but not balanced, so keep the sim's elastic band
                            on or the robot falls.
    commanded_only=True     kp = kd = 0, tau = 0: left limp.
* return_to_start() eases the commanded joints back to where they were when
  the controller connected; stop() stops publishing.

unitree_sdk2py is imported inside init(), so importing this module (for
the constants or in tests) does not need the SDK or open DDS.
"""

import math
import threading
import time

import numpy as np

from command_shaping import CommandShaper
from g1_gains import KD, KP
from g1_joint_limits import G1_29DOF_JOINT_LIMITS

G1_NUM_MOTOR = 29

# Kept for scripts that used the old names.
LEFT_SHOULDER_PITCH = 15
LEFT_SHOULDER_ROLL = 16
LEFT_ELBOW = 18
RIGHT_SHOULDER_PITCH = 22
RIGHT_SHOULDER_ROLL = 23
RIGHT_ELBOW = 25

# Upper body only: waist (12-14) and both arms (15-28). Legs need --legs.
UPPER_BODY = set(range(12, 29))
ALL_JOINTS = set(range(G1_NUM_MOTOR))

_LIMITS = {lim.index: (lim.lower, lim.upper) for lim in G1_29DOF_JOINT_LIMITS.values()}


class Mode:
    PR = 0


def validate_targets(targets, allowed):
    """{index: angle} -> cleaned dict, or ValueError. Clamps to the real limits."""
    cleaned = {}
    for index, value in targets.items():
        index = int(index)
        if index not in allowed:
            raise ValueError(f"Motor {index} is not an allowed target")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Motor {index} received a non-finite target")
        lower, upper = _LIMITS[index]
        cleaned[index] = min(max(value, lower), upper)
    return cleaned


class MujocoPoseController:
    """Publish person_id joint targets to unitree_mujoco over rt/lowcmd."""

    def __init__(self, domain_id=1, interface="lo", control_hz=500.0, max_command_speed=5.0,
                 smoothing_tau=0.05, commanded_only=False, gravity_comp=True, allow_legs=False):
        if control_hz <= 0:
            raise ValueError("control_hz must be greater than zero")
        if max_command_speed <= 0:
            raise ValueError("max_command_speed must be greater than zero")
        self.domain_id = domain_id
        self.interface = interface
        self.control_dt = 1.0 / control_hz
        self.max_command_speed = max_command_speed
        self.smoothing_tau = smoothing_tau
        self.commanded_only = commanded_only
        self.allowed = ALL_JOINTS if allow_legs else UPPER_BODY
        self.gravity = None
        if gravity_comp:
            from g1_sim import GravityComp

            self.gravity = GravityComp()
        self.low_state = None
        self.latest_q = None
        self.mode_machine = 0
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active = set()        # joints that have received a target
        self.hold_positions = None
        self.shaper = None
        self.thread = None
        self.publisher = None
        self.published = 0

    # -- DDS -----------------------------------------------------------------
    def init(self, timeout=10.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        ChannelFactoryInitialize(self.domain_id, self.interface)
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._on_low_state, 10)
        if not self.ready.wait(timeout):
            raise TimeoutError(
                "No rt/lowstate received. Check that unitree_mujoco is running "
                f"with DOMAIN_ID={self.domain_id} and INTERFACE='{self.interface}' "
                "(simulate_python/config.py).")

    def _on_low_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(G1_NUM_MOTOR)])
        with self.lock:
            self.low_state = msg
            self.latest_q = q
        if not self.ready.is_set():
            # The first simulator state is the safe start / return pose.
            self.mode_machine = msg.mode_machine
            self.hold_positions = q.copy()
            self.shaper = CommandShaper(q, self.control_dt, tau_s=self.smoothing_tau,
                                        max_speed=self.max_command_speed)
            self.ready.set()

    # -- targets ---------------------------------------------------------------
    def start(self):
        if not self.ready.is_set():
            raise RuntimeError("Call init() before start()")
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="mujoco-pose-control", daemon=True)
        self.thread.start()

    def set_targets(self, targets):
        """Partial {motor_index: angle_rad}; joints not mentioned keep their
        previous target."""
        cleaned = validate_targets(targets, self.allowed)
        with self.lock:
            for index, value in cleaned.items():
                self.shaper.set_target(index, value)
                self.active.add(index)

    def clear_targets(self):
        """Send the commanded joints back toward their start positions."""
        with self.lock:
            for index in self.active:
                self.shaper.set_target(index, self.hold_positions[index])

    def return_to_start(self, timeout=5.0, tolerance=0.02):
        """Ease commanded joints back to the start pose; True once there."""
        if self.shaper is None:
            return True
        self.clear_targets()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                q = self.shaper.q.copy()
                active = list(self.active)
            if all(abs(q[i] - self.hold_positions[i]) <= tolerance for i in active):
                return True
            time.sleep(0.02)
        return False

    def measured_positions(self):
        """Latest rt/lowstate joint positions (29,), or None."""
        with self.lock:
            return None if self.latest_q is None else self.latest_q.copy()

    def commanded_positions(self):
        with self.lock:
            return self.shaper.q.copy()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    # -- control loop ----------------------------------------------------------
    def _run(self):
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            self._write_once()
            next_tick += self.control_dt
            delay = next_tick - time.monotonic()
            if delay < -0.1:
                next_tick = time.monotonic()  # fell far behind (VM stall): resync, don't burst
            self.stop_event.wait(max(0.0, delay))

    def build_command(self):
        """One control tick -> list of (q, kp, kd, tau) per motor. Pure apart
        from advancing the shaper, so it can be tested without DDS."""
        with self.lock:
            q_cmd = self.shaper.tick()
            active = set(self.active)
            measured = self.latest_q.copy() if self.latest_q is not None else q_cmd
        tau = self.gravity(measured) if self.gravity is not None else np.zeros(G1_NUM_MOTOR)
        rows = []
        for i in range(G1_NUM_MOTOR):
            if i in active:
                rows.append((q_cmd[i], KP[i], KD[i], float(tau[i])))
            elif self.commanded_only:
                rows.append((measured[i], 0.0, 0.0, 0.0))
            else:
                rows.append((self.hold_positions[i], KP[i], KD[i], 0.0))
        return rows

    def _write_once(self):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine
        for i, (q, kp, kd, tau) in enumerate(self.build_command()):
            motor = self.low_cmd.motor_cmd[i]
            motor.mode = 1
            motor.q = float(q)
            motor.dq = 0.0
            motor.kp = float(kp)
            motor.kd = float(kd)
            motor.tau = tau
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        self.published += 1
