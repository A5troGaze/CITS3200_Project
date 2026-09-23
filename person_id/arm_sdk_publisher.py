"""
Real-robot path: drive the G1's arms and waist through rt/arm_sdk while
Unitree's own locomotion controller keeps balancing the legs.

NOT TESTED ON HARDWARE. Dry-run unless constructed with real=True (CLI:
--real), and never enabled by default.

Protocol (unitree_sdk2_python example/g1/high_level/g1_arm7_sdk_dds_example.py,
see G1_ARM_CONVENTIONS.md section 5):
* topic rt/arm_sdk, unitree_hg LowCmd_, CRC on every message;
* motor_cmd[29].q is the blend weight: 0 = locomotion controller owns the
  arms, 1 = arm_sdk does. It is ramped 0 -> 1 over ramp_s on engage and
  1 -> 0 on disengage; it never jumps;
* joints covered: waist 12-14 and arms 15-28 (never legs);
* kp 60, kd 1.5 per joint as in the example, published at 50 Hz by default
  (the example's rate). Targets go through the same CommandShaper as the sim.
* the real robot uses DDS domain 0 on its wired interface (e.g. enp3s0).

On a waist-locked G1 (29-DoF units with waist roll/pitch locked), run the
pipeline with --waist yaw so roll/pitch stay at 0.
"""

import math
import threading
import time

import numpy as np

from command_shaping import CommandShaper
from g1_gains import ARM_SDK_KD, ARM_SDK_KP
from mujoco_pose_controller import validate_targets

ARM_SDK_JOINTS = list(range(12, 29))
WEIGHT_INDEX = 29


class ArmSdkPublisher:
    def __init__(self, interface=None, domain_id=0, real=False, control_hz=50.0,
                 ramp_s=2.0, max_command_speed=3.0, smoothing_tau=0.08, log=print):
        self.real = bool(real)
        if self.real and not interface:
            raise ValueError("--real needs the robot's network interface (e.g. enp3s0)")
        self.interface = interface
        self.domain_id = domain_id
        self.dt = 1.0 / control_hz
        self.ramp_s = ramp_s
        self.max_command_speed = max_command_speed
        self.smoothing_tau = smoothing_tau
        self.log = log
        self.weight = 0.0
        self.weight_target = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread = None
        self.shaper = None
        self.start_q = None
        self.latest_q = None
        self.sent = 0

    # ---- connection ----------------------------------------------------------
    def init(self, timeout=10.0, start_q=None):
        """Real: connect and wait for rt/lowstate. Dry-run: start from start_q
        (or zeros) with no DDS at all."""
        if not self.real:
            q = np.zeros(29) if start_q is None else np.asarray(start_q, dtype=float)
            self._set_start(q)
            self.log("[arm_sdk] DRY RUN: nothing is published. Pass --real to drive a robot.")
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        ChannelFactoryInitialize(self.domain_id, self.interface)
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._on_low_state, 10)
        if not self.ready.wait(timeout):
            raise TimeoutError(f"No rt/lowstate from the robot on {self.interface} (domain {self.domain_id})")

    def _on_low_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(29)])
        with self.lock:
            self.latest_q = q
        if not self.ready.is_set():
            self._set_start(q)

    def _set_start(self, q):
        self.start_q = q.copy()
        self.shaper = CommandShaper(q, self.dt, tau_s=self.smoothing_tau, max_speed=self.max_command_speed,
                                    engage_s=self.ramp_s)
        self.ready.set()

    # ---- engage / targets ----------------------------------------------------
    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="arm-sdk", daemon=True)
        self.thread.start()

    def engage(self):
        """Ramp the arm_sdk weight to 1 (take the arms from locomotion)."""
        with self.lock:
            self.weight_target = 1.0

    def set_targets(self, targets):
        cleaned = validate_targets(targets, set(ARM_SDK_JOINTS))
        with self.lock:
            for i, v in cleaned.items():
                self.shaper.set_target(i, v)

    def disengage(self, timeout=None):
        """Ease the arms back to their start pose, then ramp the weight to 0
        (locomotion takes the arms back). Blocks until done or timeout."""
        with self.lock:
            for i in ARM_SDK_JOINTS:
                self.shaper.set_target(i, self.start_q[i])
        deadline = time.monotonic() + (timeout or (3.0 + 2 * self.ramp_s))
        while time.monotonic() < deadline:
            with self.lock:
                back = np.all(np.abs(self.shaper.q[ARM_SDK_JOINTS] - self.start_q[ARM_SDK_JOINTS]) < 0.03)
            if back:
                break
            time.sleep(0.02)
        with self.lock:
            self.weight_target = 0.0
        while time.monotonic() < deadline and self.weight > 0.0:
            time.sleep(0.02)
        return self.weight == 0.0

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    # ---- loop ----------------------------------------------------------------
    def step_weight(self):
        step = self.dt / self.ramp_s if self.ramp_s > 0 else 1.0
        self.weight = float(np.clip(self.weight + np.clip(self.weight_target - self.weight, -step, step), 0.0, 1.0))
        return self.weight

    def build_message(self):
        """One tick -> (weight, {index: q}). No DDS, testable."""
        with self.lock:
            w = self.step_weight()
            q = self.shaper.tick()
        return w, {i: float(q[i]) for i in ARM_SDK_JOINTS}

    def _run(self):
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            w, q = self.build_message()
            if self.real:
                self._publish(w, q)
            elif self.sent % int(1.0 / self.dt) == 0:
                self.log(f"[arm_sdk dry-run] weight {w:.2f} "
                         + " ".join(f"{i}:{v:+.2f}" for i, v in q.items()))
            self.sent += 1
            next_tick += self.dt
            self.stop_event.wait(max(0.0, next_tick - time.monotonic()))

    def _publish(self, weight, q):
        cmd = self.low_cmd
        cmd.motor_cmd[WEIGHT_INDEX].q = weight
        for i, value in q.items():
            if not math.isfinite(value):
                raise ValueError(f"non-finite arm_sdk target for motor {i}")
            m = cmd.motor_cmd[i]
            m.q = value
            m.dq = 0.0
            m.tau = 0.0
            m.kp = ARM_SDK_KP
            m.kd = ARM_SDK_KD
        cmd.crc = self.crc.Crc(cmd)
        self.publisher.Write(cmd)
