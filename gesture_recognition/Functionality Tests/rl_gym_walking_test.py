"""
Combines Christo's "stand on own weight without falling" fix
(simulation_controller.py, commit a479d61) with the unitree_rl_gym
pretrained leg policy, in three phases:

  Phase 1 (0 -> RAMP_DURATION seconds):
      Same as SimController's fixed ramp -- all 29 joints ease from
      whatever pose the robot spawned in (self.home_pose_, captured
      from the first low_state message) towards that SAME home pose.
      This is Christo's fix: ramping towards the robot's own natural
      pose (instead of towards a hardcoded 0.0) is what lets it support
      its own weight without the elastic band.

  Phase 2 (RAMP_DURATION -> RAMP_DURATION + SETTLE_DURATION seconds):
      Hold steady at home_pose_ for a few extra seconds so the robot is
      fully stable before anything else touches it. This is the
      "release the elastic band and let it support its own weight"
      step Christo asked for, made explicit as its own phase.

  Phase 3 (after that):
      Hand the 12 leg joints over to the unitree_rl_gym pretrained
      policy. Waist/arms stay at home_pose_ (gestures aren't wired in
      yet). The walk command (cmd) does NOT jump straight to a fixed
      speed -- it ramps up slowly from 0 to a small target speed over
      CMD_RAMP_DURATION seconds, because Christo flagged the sim's
      physics as very slippery ("walking on ice") -- any sudden
      movement makes the robot fall over.

Still NOT wired to gestures or the camera -- this is purely to validate
the combined stand+walk behaviour.

Prerequisites: same as before -- unitree_mujoco running with our G1
scene, unitree_rl_gym cloned locally, torch installed. See
SETUP_walking_policy_test.md.

NOT YET VERIFIED:
- Whether home_pose_'s leg angles are close to unitree_rl_gym's own
  DEFAULT_LEG_ANGLES. If they're very different, the Phase 2 -> Phase 3
  handover will still be a jump (the policy takes over assuming the
  legs already look like DEFAULT_LEG_ANGLES). Watch the transition
  moment specifically when testing -- if the robot stumbles exactly at
  the phase 2/3 boundary, this mismatch is why, and the handover needs
  to blend between the two poses instead of switching instantly.
"""

import os
import time

import numpy as np
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


# == Policy config, copied from unitree_rl_gym's deploy/deploy_mujoco/configs/g1.yaml ==
POLICY_PATH = os.path.expanduser(
    "~/CITS3200/Dependencies/unitree_rl_gym/deploy/pre_train/g1/motion.pt"
)

CONTROL_DT = 0.002           # matches SimController's control_dt_ (500Hz)
CONTROL_DECIMATION = 10      # policy runs every 10 control steps -> 50Hz, per g1.yaml

LEG_KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
LEG_KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
DEFAULT_LEG_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32
)

ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACTIONS = 12
NUM_OBS = 47
GAIT_PERIOD = 0.8

# -- Phase timing --
RAMP_DURATION = 3.0          # phase 1: ease to home pose (same duration as SimController's ramp_duration_)
SETTLE_DURATION = 2.0        # phase 2: hold home pose steady before handing legs to the policy
CMD_RAMP_DURATION = 5.0      # phase 3: cmd eases from 0 up to TARGET_CMD over this many seconds

# Per Christo: physics is slippery, sudden movement = fall. Keep this small.
# Tune down further (or even smaller) if it still falls during phase 3.
TARGET_CMD = np.array([0.1, 0.0, 0.0], dtype=np.float32)

G1_NUM_MOTOR = 29
LEG_INDICES = list(range(12))  # motor indices 0-11: legs (unverified assumption, see walking_policy_test's earlier notes)

# Same Kp/Kd as simulation_controller.py's SimController, used for phases
# 1 and 2 (all 29 joints) and for the non-leg joints in phase 3.
FULL_KP = [
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40
]
FULL_KD = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1
]


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class Mode:
    PR = 0


class StandThenWalkTest:
    """Phase 1+2: hold home pose (Christo's fix) on all 29 joints.
    Phase 3: hand the 12 leg joints to the pretrained policy, with cmd
    ramping up slowly. Waist/arms stay at home pose throughout (no
    gestures yet)."""

    def __init__(self):
        self.time_ = 0.0
        self.low_state = None
        self.ready_ = False
        self.mode_machine_ = 0
        self.home_pose_ = None
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.policy = torch.jit.load(POLICY_PATH)
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.target_leg_angles = DEFAULT_LEG_ANGLES.copy()
        self.policy_counter = 0

    def init(self):
        ChannelFactoryInitialize(1, "lo")
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_low_state, 10)

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        if not self.ready_:
            self.mode_machine_ = msg.mode_machine
            self.home_pose_ = [msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            self.ready_ = True
            for i in range(G1_NUM_MOTOR):
                print(f"[home pose] joint {i:2d}: q = {msg.motor_state[i].q:+.4f}")

    def start(self):
        while not self.ready_:
            time.sleep(0.1)
        self.thread_ = RecurrentThread(interval=CONTROL_DT, target=self._write, name="stand_then_walk_test")
        self.thread_.Start()

    def _run_policy_step(self):
        """One policy inference, updating self.target_leg_angles. Only
        called during phase 3, at 50Hz (every CONTROL_DECIMATION steps)."""
        self.policy_counter += 1
        if self.policy_counter % CONTROL_DECIMATION != 0:
            return

        qj = np.array([self.low_state.motor_state[i].q for i in LEG_INDICES], dtype=np.float32)
        dqj = np.array([self.low_state.motor_state[i].dq for i in LEG_INDICES], dtype=np.float32)
        quat = self.low_state.imu_state.quaternion         # [w, x, y, z] -- confirmed against unitree_hg IDL
        omega = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        qj_obs = (qj - DEFAULT_LEG_ANGLES) * DOF_POS_SCALE
        dqj_obs = dqj * DOF_VEL_SCALE
        gravity = get_gravity_orientation(quat)
        omega_obs = omega * ANG_VEL_SCALE

        # cmd ramps from 0 up to TARGET_CMD over CMD_RAMP_DURATION seconds,
        # measured from the start of phase 3 (not from t=0 overall).
        time_in_phase3 = self.time_ - (RAMP_DURATION + SETTLE_DURATION)
        cmd_ratio = np.clip(time_in_phase3 / CMD_RAMP_DURATION, 0.0, 1.0)
        cmd = TARGET_CMD * cmd_ratio

        period = GAIT_PERIOD
        phase = (self.time_ % period) / period
        sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

        obs = np.zeros(NUM_OBS, dtype=np.float32)
        obs[0:3] = omega_obs
        obs[3:6] = gravity
        obs[6:9] = cmd * CMD_SCALE
        obs[9:21] = qj_obs
        obs[21:33] = dqj_obs
        obs[33:45] = self.action
        obs[45:47] = [sin_phase, cos_phase]

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.target_leg_angles = self.action * ACTION_SCALE + DEFAULT_LEG_ANGLES

    def _write(self):
        self.time_ += CONTROL_DT
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        if self.time_ < RAMP_DURATION:
            # -- Phase 1: ease from current pose to home pose (Christo's fix) --
            ratio = np.clip(self.time_ / RAMP_DURATION, 0.0, 1.0)
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = (
                    (1.0 - ratio) * self.low_state.motor_state[i].q + ratio * self.home_pose_[i]
                )
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = FULL_KP[i]
                self.low_cmd.motor_cmd[i].kd = FULL_KD[i]

        elif self.time_ < RAMP_DURATION + SETTLE_DURATION:
            # -- Phase 2: hold home pose steady, let it fully stabilise --
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = self.home_pose_[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = FULL_KP[i]
                self.low_cmd.motor_cmd[i].kd = FULL_KD[i]

        else:
            # -- Phase 3: legs -> RL policy (cmd ramping up), waist/arms -> home pose --
            self._run_policy_step()

            for idx_in_leg, motor_i in enumerate(LEG_INDICES):
                self.low_cmd.motor_cmd[motor_i].mode = 1
                self.low_cmd.motor_cmd[motor_i].tau = 0.0
                self.low_cmd.motor_cmd[motor_i].q = float(self.target_leg_angles[idx_in_leg])
                self.low_cmd.motor_cmd[motor_i].dq = 0.0
                self.low_cmd.motor_cmd[motor_i].kp = float(LEG_KP[idx_in_leg])
                self.low_cmd.motor_cmd[motor_i].kd = float(LEG_KD[idx_in_leg])

            for motor_i in range(12, G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[motor_i].mode = 1
                self.low_cmd.motor_cmd[motor_i].tau = 0.0
                self.low_cmd.motor_cmd[motor_i].q = self.home_pose_[motor_i]
                self.low_cmd.motor_cmd[motor_i].dq = 0.0
                self.low_cmd.motor_cmd[motor_i].kp = FULL_KP[motor_i]
                self.low_cmd.motor_cmd[motor_i].kd = FULL_KD[motor_i]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == "__main__":
    controller = StandThenWalkTest()
    controller.init()
    controller.start()
    print(
        f"Running: phase 1 (0-{RAMP_DURATION}s) stand up, "
        f"phase 2 ({RAMP_DURATION}-{RAMP_DURATION + SETTLE_DURATION}s) settle, "
        f"phase 3 (after) slow walk ramping to {TARGET_CMD.tolist()}. Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
