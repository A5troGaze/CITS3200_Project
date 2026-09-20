"""
Validation-only script: drives ONLY the G1's 12 leg joints using the
pretrained unitree_rl_gym locomotion policy (deploy/pre_train/g1/motion.pt),
over the SAME DDS/unitree_mujoco simulation that SimController normally
talks to. The other 17 joints (waist + both arms) are held at a fixed
standing pose -- this does NOT touch gesture recognition or camera input.

Purpose: the pretrained policy was trained against unitree_rl_gym's own
G1 MJCF (deploy/deploy_mujoco/configs/g1.yaml -> g1_description/scene.xml),
which may or may not be kinematically identical (same joint order) to the
G1 model our team's unitree_mujoco actually loads. This script is the
cheapest way to find out: if the robot stands and walks normally, motor
indices 0-11 really are "left leg x6, right leg x6" in the order the
policy expects, and it is safe to build the full gesture+walking
SimController on top of this. If the robot immediately falls over or
its legs move incoherently, the joint order does NOT match and the
mapping needs to be re-derived before any further work.

Prerequisites:
- unitree_mujoco is already running separately, loaded with our team's
  G1 scene (the same simulation SimController normally connects to).
- unitree_rl_gym has been cloned locally (see POLICY_PATH below) so the
  pretrained weights are available.
- PyTorch is installed in the environment this script runs in.

NOT YET VERIFIED (check before / while running):
- The field names `self.low_state.imu_state.quaternion` and
  `.gyroscope` are my best guess at the standard unitree_hg LowState_
  IDL layout, based on how other Unitree G1 SDK examples read IMU data.
  If this errors or the names differ, check the actual
  unitree_hg_msg_dds__LowState_ definition (e.g.
  `print(dir(low_state.imu_state))` inside _on_low_state) and fix the
  two field accesses in _write() accordingly.

Usage:
    python walking_policy_test.py
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


# == Config, copied from unitree_rl_gym's deploy/deploy_mujoco/configs/g1.yaml ==
# Adjust this path if you cloned unitree_rl_gym somewhere else.
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

# Fixed test command: walk forward slowly. This is what deploy_mujoco.py's
# cmd_init does too. Swap this for gesture-driven input in a later step,
# once this validation passes.
CMD = np.array([0.1, 0.0, 0.0], dtype=np.float32)

G1_NUM_MOTOR = 29
LEG_INDICES = list(range(12))  # motor indices 0-11: legs. THIS is the assumption being tested.

# Same Kp/Kd used for the non-leg joints in simulation_controller.py's
# SimController, so the upper body holds a normal standing pose instead
# of going limp while we're only testing the legs.
UPPER_KP = [
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
]
UPPER_KD = [1] * 17


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class Mode:
    PR = 0


class WalkingPolicyTest:
    """Drives only the leg joints via the pretrained policy; holds the
    waist/arms at a fixed pose. Not wired to gestures yet -- purely to
    validate the policy transfers onto our team's G1 model."""

    def __init__(self):
        self.low_state = None
        self.ready_ = False
        self.mode_machine_ = 0
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.policy = torch.jit.load(POLICY_PATH)
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.target_leg_angles = DEFAULT_LEG_ANGLES.copy()
        self.counter = 0

    def init(self):
        ChannelFactoryInitialize(1, "lo")
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_low_state, 10)

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        if not self.ready_:
            self.mode_machine_ = self.low_state.mode_machine
            self.ready_ = True

    def start(self):
        while not self.ready_:
            time.sleep(0.1)
        self.thread_ = RecurrentThread(interval=CONTROL_DT, target=self._write, name="walking_policy_test")
        self.thread_.Start()

    def _write(self):
        self.counter += 1

        # -- Run the policy every 10 control steps (50Hz), same as deploy_mujoco.py --
        if self.counter % CONTROL_DECIMATION == 0:
            qj = np.array([self.low_state.motor_state[i].q for i in LEG_INDICES], dtype=np.float32)
            dqj = np.array([self.low_state.motor_state[i].dq for i in LEG_INDICES], dtype=np.float32)
            # NOT YET VERIFIED -- see module docstring.
            quat = self.low_state.imu_state.quaternion       # expected [w, x, y, z]
            omega = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

            qj_obs = (qj - DEFAULT_LEG_ANGLES) * DOF_POS_SCALE
            dqj_obs = dqj * DOF_VEL_SCALE
            gravity = get_gravity_orientation(quat)
            omega_obs = omega * ANG_VEL_SCALE

            count = self.counter * CONTROL_DT
            phase = (count % GAIT_PERIOD) / GAIT_PERIOD
            sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

            obs = np.zeros(NUM_OBS, dtype=np.float32)
            obs[0:3] = omega_obs
            obs[3:6] = gravity
            obs[6:9] = CMD * CMD_SCALE
            obs[9:21] = qj_obs
            obs[21:33] = dqj_obs
            obs[33:45] = self.action
            obs[45:47] = [sin_phase, cos_phase]

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            self.action = self.policy(obs_tensor).detach().numpy().squeeze()
            self.target_leg_angles = self.action * ACTION_SCALE + DEFAULT_LEG_ANGLES

            if self.counter % 25 == 0:   # print roughly twice a second, not every single 50Hz policy step
                print("action:", self.action)
                print("target_leg_angles:", self.target_leg_angles)

        # -- PD control every control step (500Hz) --
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        for idx_in_leg, motor_i in enumerate(LEG_INDICES):
            self.low_cmd.motor_cmd[motor_i].mode = 1
            self.low_cmd.motor_cmd[motor_i].tau = 0.0
            self.low_cmd.motor_cmd[motor_i].q = float(self.target_leg_angles[idx_in_leg])
            self.low_cmd.motor_cmd[motor_i].dq = 0.0
            self.low_cmd.motor_cmd[motor_i].kp = float(LEG_KP[idx_in_leg])
            self.low_cmd.motor_cmd[motor_i].kd = float(LEG_KD[idx_in_leg])

        for offset, motor_i in enumerate(range(12, G1_NUM_MOTOR)):
            self.low_cmd.motor_cmd[motor_i].mode = 1
            self.low_cmd.motor_cmd[motor_i].tau = 0.0
            self.low_cmd.motor_cmd[motor_i].q = 0.0
            self.low_cmd.motor_cmd[motor_i].dq = 0.0
            self.low_cmd.motor_cmd[motor_i].kp = UPPER_KP[offset]
            self.low_cmd.motor_cmd[motor_i].kd = UPPER_KD[offset]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == "__main__":
    controller = WalkingPolicyTest()
    controller.init()
    controller.start()
    print("Walking policy test running against unitree_mujoco. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
