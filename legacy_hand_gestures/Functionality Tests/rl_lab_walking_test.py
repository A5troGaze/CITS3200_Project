"""
Standalone test: drive the G1 using unitree_rl_lab's pretrained full-body
(29-joint) velocity policy (policy.onnx), reverse-engineered from the
repo's own C++ deploy code since there is no Python reference:

  - deploy/robots/g1_29dof/src/State_RLBase.cpp
      confirms action[i] (policy output, "policy order") is written to
      motor_cmd[joint_ids_map[i]] (real SDK motor index). We apply the
      same mapping symmetrically when reading joint state for the
      observation (policy order slot i <- motor_state[joint_ids_map[i]]).

  - deploy/include/isaaclab/envs/mdp/observations/observations.h
      defines each single-frame observation term.

  - deploy/include/isaaclab/manager/observation_manager.h
    + deploy/include/isaaclab/manager/manager_term_cfg.h
      confirm each term keeps its OWN rolling buffer of `history_length`
      frames (oldest at front, newest at back), and -- since our
      deploy.yaml does not set `use_gym_history` (defaults to false) --
      the final observation is built by concatenating each term's full
      5-frame buffer one term at a time (NOT interleaved by timestep):

          [ang_vel x5, gravity x5, cmd x5, joint_pos_rel x5,
           joint_vel_rel x5, last_action x5]   (96 * 5 = 480 dims)

      Per-frame values are clipped (none configured here) then scaled
      by each term's `scale` array before going into the buffer.

Phase 1-2 now ease to and hold DEFAULT_JOINT_POS (converted into real
motor order) instead of the robot's arbitrary connect-time home_pose_.
Reasoning: the policy's joint_pos_rel term is (q - DEFAULT_JOINT_POS),
so the policy expects to take over from a pose where that term is
near zero. Handing off from an unrelated home_pose_ instead feeds it
an out-of-distribution starting observation, which matched the
symptom seen (a handful of plausible-looking action steps, then a
sudden divergent spike). home_pose_ is still captured for reference
but is no longer used as a control target.

Phase 3 still hands ALL 29 joints to the policy -- NOT just the legs,
because this policy controls the whole body (legs + waist + arms) at
once. That means, as currently written, gestures CANNOT be layered on
top during phase 3 -- the policy owns the arms while walking. This is
the architecture question that still needs a team decision; this
script intentionally does not attempt to resolve it, so the walking
behaviour itself can be validated in isolation first.

NOT YET VERIFIED:
- The policy.onnx input/output tensor names and shapes -- run
  print_onnx_io() (below) once before the first real test to confirm
  they match the 480-in / 29-out assumption this script makes.
- Whether `env->robot->data.joint_pos` (used to build joint_pos_rel /
  joint_vel_rel in the C++ observation code) is already in "policy
  order" (so default_joint_pos/joint_ids_map line up with it directly,
  as assumed here) or in raw SDK motor order requiring an extra
  translation step we don't have visibility into (that logic lives in
  unitree_articulation.h / unitree_rl_lab's BaseArticulation class,
  which we have not read). If the robot's motion looks systematically
  wrong (not just unstable, but wrong-looking, e.g. knees bending
  backwards) this mapping assumption is the first thing to check.

Prerequisites: unitree_mujoco running with our G1 scene, unitree_rl_lab
cloned locally, onnxruntime installed (`pip install onnxruntime`).
"""

import os
import time
from collections import deque

import numpy as np
import onnxruntime as ort

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


# == Config, copied from unitree_rl_lab's ==
# == deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml ==
POLICY_PATH = os.path.expanduser(
    "~/CITS3200/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"
)

JOINT_IDS_MAP = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
                 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]  # policy-order slot i -> real motor index

CONTROL_DT = 0.002          # SimController's control_dt_ (500Hz)
STEP_DT = 0.02              # policy's own step_dt (50Hz) -> matches CONTROL_DECIMATION below
CONTROL_DECIMATION = round(STEP_DT / CONTROL_DT)  # = 10

NUM_JOINTS = 29
HISTORY_LENGTH = 5

STIFFNESS = np.array(
    [100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
     200.0, 200.0, 200.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
     40.0, 40.0, 40.0, 40.0, 40.0], dtype=np.float32)
DAMPING = np.array(
    [2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 5.0, 5.0, 5.0,
     10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    dtype=np.float32)
DEFAULT_JOINT_POS = np.array(
    [-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2,
     0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32)  # policy order, NOT the same order as our SimController's home_pose_

# DEFAULT_JOINT_POS is in policy order; convert to real motor order so
# it can be used directly against motor_cmd[i]/motor_state[i] during
# phases 1-2, instead of the robot's arbitrary connect-time home_pose_.
DEFAULT_JOINT_POS_MOTOR_ORDER = np.zeros(NUM_JOINTS, dtype=np.float32)
for _slot, _motor_i in enumerate(JOINT_IDS_MAP):
    DEFAULT_JOINT_POS_MOTOR_ORDER[_motor_i] = DEFAULT_JOINT_POS[_slot]

ACTION_SCALE = np.full(NUM_JOINTS, 0.25, dtype=np.float32)

# Per-term scale, in the order the terms appear in deploy.yaml's observations:
ANG_VEL_SCALE = np.array([0.2, 0.2, 0.2], dtype=np.float32)
GRAVITY_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
CMD_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
JOINT_POS_SCALE = np.full(NUM_JOINTS, 1.0, dtype=np.float32)
JOINT_VEL_SCALE = np.full(NUM_JOINTS, 0.05, dtype=np.float32)
LAST_ACTION_SCALE = np.full(NUM_JOINTS, 1.0, dtype=np.float32)

CMD_RANGE = {"lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.3, 0.3), "ang_vel_z": (-0.2, 0.2)}

# -- Phase timing (same pattern as walking_policy_test.py) --
RAMP_DURATION = 3.0
SETTLE_DURATION = 2.0
CMD_RAMP_DURATION = 5.0
# Per Christo: physics is slippery. Start small; this is a first guess.
TARGET_CMD = np.array([0.1, 0.0, 0.0], dtype=np.float32)

G1_NUM_MOTOR = 29


def get_projected_gravity(quaternion):
    """Same formula as before -- gravity vector expressed in the body frame."""
    qw, qx, qy, qz = quaternion
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def print_onnx_io(path=POLICY_PATH):
    """Run this once before the first real test to sanity-check the
    policy's actual input/output shapes against what this script
    assumes (480 in, 29 out)."""
    sess = ort.InferenceSession(path)
    for i in sess.get_inputs():
        print("input:", i.name, i.shape)
    for o in sess.get_outputs():
        print("output:", o.name, o.shape)


class Mode:
    PR = 0


class TermHistory:
    """Mirrors ObservationTermCfg's buffer: a deque of up to
    HISTORY_LENGTH frames, oldest at the front. get() concatenates the
    whole buffer oldest->newest, matching term.get() in
    manager_term_cfg.h when use_gym_history is false."""

    def __init__(self, dim):
        self.dim = dim
        self.buf = deque(maxlen=HISTORY_LENGTH)

    def reset(self, frame):
        self.buf.clear()
        for _ in range(HISTORY_LENGTH):
            self.buf.append(frame.copy())

    def add(self, frame):
        self.buf.append(frame.copy())

    def get(self):
        return np.concatenate(list(self.buf))  # oldest -> newest


class RLLabWalkingTest:
    """Phase 1+2: ease to and hold DEFAULT_JOINT_POS (the policy's own
    expected starting pose, converted to motor order) -- not the
    robot's arbitrary home_pose_. Phase 3: hand ALL 29 joints to the
    unitree_rl_lab policy, cmd ramping up slowly."""

    def __init__(self):
        self.time_ = 0.0
        self.low_state = None
        self.ready_ = False
        self.mode_machine_ = 0
        self.home_pose_ = None  # captured for reference only; not used as a control target anymore
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.session = ort.InferenceSession(POLICY_PATH)
        self.input_name = self.session.get_inputs()[0].name

        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.target_joint_pos = DEFAULT_JOINT_POS.copy()
        self.policy_counter = 0

        self.term_ang_vel = TermHistory(3)
        self.term_gravity = TermHistory(3)
        self.term_cmd = TermHistory(3)
        self.term_joint_pos_rel = TermHistory(NUM_JOINTS)
        self.term_joint_vel_rel = TermHistory(NUM_JOINTS)
        self.term_last_action = TermHistory(NUM_JOINTS)
        self._history_initialised = False

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

    def start(self):
        while not self.ready_:
            time.sleep(0.1)
        self.thread_ = RecurrentThread(interval=CONTROL_DT, target=self._write, name="rl_lab_walking_test")
        self.thread_.Start()

    def _read_policy_order_joint_state(self):
        """policy-order slot i <- motor_state[JOINT_IDS_MAP[i]], per
        State_RLBase.cpp's (inverse) mapping convention."""
        q = np.array([self.low_state.motor_state[JOINT_IDS_MAP[i]].q for i in range(NUM_JOINTS)], dtype=np.float32)
        dq = np.array([self.low_state.motor_state[JOINT_IDS_MAP[i]].dq for i in range(NUM_JOINTS)], dtype=np.float32)
        return q, dq

    def _run_policy_step(self):
        self.policy_counter += 1
        if self.policy_counter % CONTROL_DECIMATION != 0:
            return

        q, dq = self._read_policy_order_joint_state()
        quat = self.low_state.imu_state.quaternion
        omega = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        ang_vel_frame = omega * ANG_VEL_SCALE
        gravity_frame = get_projected_gravity(quat) * GRAVITY_SCALE

        time_in_phase3 = self.time_ - (RAMP_DURATION + SETTLE_DURATION)
        cmd_ratio = np.clip(time_in_phase3 / CMD_RAMP_DURATION, 0.0, 1.0)
        cmd_frame = (TARGET_CMD * cmd_ratio) * CMD_SCALE

        joint_pos_rel_frame = (q - DEFAULT_JOINT_POS) * JOINT_POS_SCALE
        joint_vel_rel_frame = dq * JOINT_VEL_SCALE
        last_action_frame = self.last_action * LAST_ACTION_SCALE

        if not self._history_initialised:
            self.term_ang_vel.reset(ang_vel_frame)
            self.term_gravity.reset(gravity_frame)
            self.term_cmd.reset(cmd_frame)
            self.term_joint_pos_rel.reset(joint_pos_rel_frame)
            self.term_joint_vel_rel.reset(joint_vel_rel_frame)
            self.term_last_action.reset(last_action_frame)
            self._history_initialised = True
        else:
            self.term_ang_vel.add(ang_vel_frame)
            self.term_gravity.add(gravity_frame)
            self.term_cmd.add(cmd_frame)
            self.term_joint_pos_rel.add(joint_pos_rel_frame)
            self.term_joint_vel_rel.add(joint_vel_rel_frame)
            self.term_last_action.add(last_action_frame)

        # Term-block order, matching deploy.yaml's `observations:` order.
        obs = np.concatenate([
            self.term_ang_vel.get(),
            self.term_gravity.get(),
            self.term_cmd.get(),
            self.term_joint_pos_rel.get(),
            self.term_joint_vel_rel.get(),
            self.term_last_action.get(),
        ]).astype(np.float32)

        obs = obs.reshape(1, -1)  # add batch dim; verify against print_onnx_io() output
        action = self.session.run(None, {self.input_name: obs})[0].squeeze()
        print("action:", action)   # diagnostic -- remove once behaviour looks right

        self.last_action = action.astype(np.float32)
        self.target_joint_pos = action * ACTION_SCALE + DEFAULT_JOINT_POS

    def _write(self):
        self.time_ += CONTROL_DT
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        if self.time_ < RAMP_DURATION:
            ratio = np.clip(self.time_ / RAMP_DURATION, 0.0, 1.0)
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = (
                    (1.0 - ratio) * self.low_state.motor_state[i].q + ratio * DEFAULT_JOINT_POS_MOTOR_ORDER[i]
                )
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 60.0
                self.low_cmd.motor_cmd[i].kd = 1.5

        elif self.time_ < RAMP_DURATION + SETTLE_DURATION:
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = DEFAULT_JOINT_POS_MOTOR_ORDER[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 60.0
                self.low_cmd.motor_cmd[i].kd = 1.5

        else:
            # Phase 3: the policy owns ALL 29 joints -- no room for
            # gestures here yet, see module docstring.
            self._run_policy_step()
            for i in range(NUM_JOINTS):
                motor_i = JOINT_IDS_MAP[i]
                self.low_cmd.motor_cmd[motor_i].mode = 1
                self.low_cmd.motor_cmd[motor_i].tau = 0.0
                self.low_cmd.motor_cmd[motor_i].q = float(self.target_joint_pos[i])
                self.low_cmd.motor_cmd[motor_i].dq = 0.0
                self.low_cmd.motor_cmd[motor_i].kp = float(STIFFNESS[motor_i])
                self.low_cmd.motor_cmd[motor_i].kd = float(DAMPING[motor_i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


if __name__ == "__main__":
    print("Checking policy.onnx input/output shapes (expect 480 in, 29 out):")
    print_onnx_io()

    controller = RLLabWalkingTest()
    controller.init()
    controller.start()
    print(
        f"Running: phase 1 (0-{RAMP_DURATION}s) stand up, "
        f"phase 2 ({RAMP_DURATION}-{RAMP_DURATION + SETTLE_DURATION}s) settle, "
        f"phase 3 (after) whole-body walk ramping cmd to {TARGET_CMD.tolist()}. Ctrl+C to stop."
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass