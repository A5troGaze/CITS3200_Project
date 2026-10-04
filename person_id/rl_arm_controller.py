"""
Sim controller: unitree_rl_lab's velocity policy balances the G1 (legs and
waist), and person_id drives the arms. One rt/lowcmd publisher, so nothing
fights.

Built the way the gesture sub-team runs the policy
(Walking-Policy-Test: gesture_recognition/Functionality Tests/rl_lab_walking_test.py;
not modified):
  phase 1 (0-3 s)  ease every joint from where it is to the policy's default
                   pose, kp 60 / kd 1.5;
  phase 2 (3-5 s)  hold the default pose;
  phase 3          policy at 50 Hz (rl_lab_policy.RlLabPolicy), deploy.yaml
                   stiffness/damping, velocity command (0, 0, 0): stand in place.
The person_id change is in phase 3: arm joints 15-28 take person_id's
targets (speed-limited by command_shaping.CommandShaper) instead of the
policy's, and the policy is shown its arms at its default pose
(rl_lab_policy arm_obs="default"; see that module for why and for the
limits measured). Arm targets that arrive before phase 3 are kept and
applied from phase 3.

In the sim, start unitree_mujoco with the elastic band on (or
`sim_standing.py --band`) and press 9 to release the band once the log says
the policy is balancing. When person_id stops, the arms ease back to the
policy's default pose first; after that nothing balances the robot, so
catch it with the band (9) before quitting, or let it fall.

Same interface as MujocoPoseController (init/start/set_targets/
clear_targets/return_to_start/stop/measured_positions), so leader_pose.py
and person_id_replay.py can use either. Simulator only; on the real robot
use --real (rt/arm_sdk under Unitree's own locomotion).
"""

import math
import threading
import time

import numpy as np

from command_shaping import CommandShaper
from mujoco_pose_controller import Mode, validate_targets
from rl_lab_policy import ARM_MOTORS, DAMPING, DEFAULT_POSE_MOTOR, STEP_DT, STIFFNESS, RlLabPolicy

G1_NUM_MOTOR = 29
RAMP_S = 3.0
SETTLE_S = 2.0
STAND_KP, STAND_KD = 60.0, 1.5     # rl_lab_walking_test.py phases 1-2


class RlArmController:
    def __init__(self, domain_id=1, interface="lo", control_hz=200.0, arm_speed=1.0,
                 arm_obs="default", cmd=(0.0, 0.0, 0.0), log=print):
        self.domain_id = domain_id
        self.interface = interface
        self.control_dt = 1.0 / control_hz
        self.decimation = max(1, round(STEP_DT / self.control_dt))
        self.arm_speed = arm_speed
        self.cmd = tuple(cmd)
        self.log = log
        self.policy = RlLabPolicy(arm_obs=arm_obs)
        from g1_sim import GravityComp

        # Gravity feed-forward on the arm joints only (the policy's own joints
        # get none, as in training).
        self.gravity = GravityComp(floating=True)
        self.allowed = set(ARM_MOTORS)
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.thread = None
        self.publisher = None
        self.latest = None            # (q, dq, quat, gyro) from rt/lowstate
        self.start_q = None
        self.hold_positions = DEFAULT_POSE_MOTOR.copy()   # "home" for the arms
        self.arm_shaper = None
        self.active = set()
        self.t = 0.0
        self.ticks = 0
        self.policy_targets = DEFAULT_POSE_MOTOR.copy()
        self.announced = False
        self.published = 0
        self.mode_machine = 0

    # -- DDS ---------------------------------------------------------------
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
            raise TimeoutError("No rt/lowstate received: is the simulator running "
                               f"(domain {self.domain_id}, interface {self.interface})?")

    def _on_low_state(self, msg):
        state = (
            np.array([msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]),
            np.array([msg.motor_state[i].dq for i in range(G1_NUM_MOTOR)]),
            np.array(msg.imu_state.quaternion, dtype=float),
            np.array(msg.imu_state.gyroscope, dtype=float),
        )
        with self.lock:
            self.latest = state
        if not self.ready.is_set():
            self.mode_machine = msg.mode_machine
            self.on_first_state(state[0])
            self.ready.set()

    def on_first_state(self, q):
        self.start_q = q.copy()
        self.arm_shaper = CommandShaper(DEFAULT_POSE_MOTOR, self.control_dt, max_speed=self.arm_speed,
                                        engage_s=1.0)

    # -- targets -------------------------------------------------------------
    def start(self):
        if not self.ready.is_set():
            raise RuntimeError("Call init() before start()")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="rl-arm-control", daemon=True)
        self.thread.start()

    def set_targets(self, targets):
        """{arm motor index: angle}. Waist/leg indices are rejected: the policy owns them."""
        cleaned = validate_targets(targets, self.allowed)
        with self.lock:
            for i, v in cleaned.items():
                self.arm_shaper.set_target(i, v)
                self.active.add(i)

    def clear_targets(self):
        with self.lock:
            for i in ARM_MOTORS:
                self.arm_shaper.set_target(i, DEFAULT_POSE_MOTOR[i])

    def return_to_start(self, timeout=6.0, tolerance=0.03):
        """Ease the arms back to the policy's default pose (the policy keeps
        balancing meanwhile)."""
        if self.arm_shaper is None:
            return True
        self.clear_targets()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                q = self.arm_shaper.q.copy()
            if np.all(np.abs(q[ARM_MOTORS] - DEFAULT_POSE_MOTOR[ARM_MOTORS]) <= tolerance):
                return True
            time.sleep(0.02)
        return False

    def measured_positions(self):
        with self.lock:
            return None if self.latest is None else self.latest[0].copy()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.log("Controller stopped: nothing is balancing the robot now (press 9 in the sim to catch it).")

    # -- control -------------------------------------------------------------
    def phase(self):
        if self.t < RAMP_S:
            return 1
        if self.t < RAMP_S + SETTLE_S:
            return 2
        return 3

    def build_command(self):
        """One control tick -> (q, kp, kd, tau) arrays in motor order. DDS-free."""
        self.t += self.control_dt
        self.ticks += 1
        with self.lock:
            q, dq, quat, gyro = self.latest
            arm_q = self.arm_shaper.tick() if self.phase() == 3 else self.arm_shaper.q.copy()
        phase = self.phase()
        if phase == 1:
            ratio = min(1.0, self.t / RAMP_S)
            q_cmd = (1.0 - ratio) * self.start_q + ratio * DEFAULT_POSE_MOTOR
            return q_cmd, np.full(G1_NUM_MOTOR, STAND_KP), np.full(G1_NUM_MOTOR, STAND_KD), np.zeros(G1_NUM_MOTOR)
        if phase == 2:
            return (DEFAULT_POSE_MOTOR.copy(), np.full(G1_NUM_MOTOR, STAND_KP), np.full(G1_NUM_MOTOR, STAND_KD),
                    np.zeros(G1_NUM_MOTOR))
        if not self.announced:
            self.announced = True
            self.log("rl_lab policy is balancing the robot: release the elastic band now (key 9 in the sim).")
        if self.ticks % self.decimation == 0:
            override = {i: float(arm_q[i]) for i in ARM_MOTORS}
            self.policy_targets = self.policy.step(q, dq, quat, gyro, self.cmd, override)
        q_cmd = self.policy_targets.copy()
        q_cmd[ARM_MOTORS] = arm_q[ARM_MOTORS]
        tau = np.zeros(G1_NUM_MOTOR)
        tau[ARM_MOTORS] = self.gravity(q, quat)[ARM_MOTORS]
        return q_cmd, STIFFNESS.copy(), DAMPING.copy(), tau

    def _run(self):
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            self._write(*self.build_command())
            next_tick += self.control_dt
            delay = next_tick - time.monotonic()
            if delay < -0.1:
                next_tick = time.monotonic()
            self.stop_event.wait(max(0.0, delay))

    def _write(self, q_cmd, kp, kd, tau):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(G1_NUM_MOTOR):
            if not math.isfinite(q_cmd[i]):
                raise ValueError(f"non-finite command for motor {i}")
            m = self.low_cmd.motor_cmd[i]
            m.mode = 1
            m.q = float(q_cmd[i])
            m.dq = 0.0
            m.kp = float(kp[i])
            m.kd = float(kd[i])
            m.tau = float(tau[i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        self.published += 1
