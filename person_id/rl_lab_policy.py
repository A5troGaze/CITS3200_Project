"""
unitree_rl_lab's pretrained G1 velocity policy (policy.onnx), run from Python,
following the gesture sub-team's implementation
(Walking-Policy-Test: gesture_recognition/Functionality Tests/rl_lab_walking_test.py).
That code was not modified. The same logic is kept here without DDS, so the
person_id controller and the offline tests share one copy:

* config from unitree_rl_lab deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml
  (joint_ids_map, default_joint_pos and action offset in policy order;
  stiffness/damping in SDK motor order; action scale 0.25; step_dt 0.02 s);
* observation = 6 terms (ang vel x0.2, projected gravity, velocity command,
  joint_pos_rel, joint_vel_rel x0.05, last action), each with its own 5-frame
  history, concatenated term by term: 480 values (observation_manager.h,
  manager_term_cfg.h with use_gym_history false);
* action i -> motor_cmd[joint_ids_map[i]].q = action*0.25 + default (State_RLBase.cpp,
  joint_actions.h).

The rl_lab_walking_test.py docstring left one thing open: whether the
observation's joint_pos is in policy order. unitree_rl_lab's
deploy/include/unitree_articulation.h settles it: BaseArticulation::update()
fills joint_pos[i] = motor_state[joint_ids_map[i]], i.e. policy order, as
that script assumed.

person_id addition: arm_override. The policy still outputs all 29 joints,
but the arm joints are driven by person_id. What the policy is told about
those arms (arm_obs) matters a lot. Offline, in unitree_mujoco's G1 model
with the policy balancing and 10 static arm poses held for 10 s at 2 rad/s:
  "real"    (measured arm state):                    2/10 stayed up
  "default" (arms shown at the policy's default,     9/10 stayed up
             zero velocity; the default here)
  "phantom" (arms shown following the policy's own   similar to "default"
             commands)
Arms overhead only stays up at <= 1 rad/s, and a fast sequence of large
arm motions (T-pose -> arms forward -> overhead) still knocks the robot
over. The policy was trained for walking with its own arm swing, and
sees person_id's arms only as an unexplained push.
"""

import os
from collections import deque

import numpy as np

RL_LAB_DIR = os.path.expanduser(os.environ.get("UNITREE_RL_LAB_DIR", "~/CITS3200/Dependencies/unitree_rl_lab"))
POLICY_DIR = os.path.join(RL_LAB_DIR, "deploy", "robots", "g1_29dof", "config", "policy", "velocity", "v0")
POLICY_PATH = os.path.join(POLICY_DIR, "exported", "policy.onnx")
DEPLOY_YAML = os.path.join(POLICY_DIR, "params", "deploy.yaml")

NUM_JOINTS = 29
HISTORY_LENGTH = 5
STEP_DT = 0.02
ARM_MOTORS = list(range(15, 29))

# deploy.yaml values (checked against the file by tests/test_rl_lab_policy.py).
JOINT_IDS_MAP = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
                          17, 24, 18, 25, 19, 26, 20, 27, 21, 28])   # policy slot i -> motor index
STIFFNESS = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
                      200.0, 200.0, 200.0] + [40.0] * 14)             # motor order
DAMPING = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
                    5.0, 5.0, 5.0] + [10.0] * 14)                     # motor order
DEFAULT_JOINT_POS = np.array([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2,
                              0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0])
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05

DEFAULT_POSE_MOTOR = np.zeros(NUM_JOINTS)
DEFAULT_POSE_MOTOR[JOINT_IDS_MAP] = DEFAULT_JOINT_POS
# Motor index -> policy slot.
MOTOR_TO_SLOT = np.argsort(JOINT_IDS_MAP)


def projected_gravity(quat_wxyz):
    """Gravity direction in the body frame, q.conjugate() * (0, 0, -1)
    (unitree_articulation.h); same formula as rl_lab_walking_test.py."""
    qw, qx, qy, qz = quat_wxyz
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


class _TermHistory:
    """ObservationTermCfg's buffer: HISTORY_LENGTH frames, oldest first."""

    def __init__(self):
        self.buf = deque(maxlen=HISTORY_LENGTH)

    def add(self, frame):
        if not self.buf:
            for _ in range(HISTORY_LENGTH):
                self.buf.append(frame.copy())
        else:
            self.buf.append(frame.copy())

    def get(self):
        return np.concatenate(list(self.buf))


class RlLabPolicy:
    """One policy step every STEP_DT: step(q, dq, quat, gyro, cmd) -> motor-order joint targets."""

    def __init__(self, policy_path=POLICY_PATH, arm_obs="default", arm_last_action=True):
        """arm_obs: "real" feeds the policy the arms' measured state;
        "default" shows it the arms at its default pose with zero velocity
        whenever they are overridden (it then only feels them through the
        body's motion). arm_last_action: rewrite the arm entries of
        last_action to match the overridden targets."""
        if arm_obs not in ("real", "default", "phantom"):
            raise ValueError(f"unknown arm_obs {arm_obs!r}")
        self.arm_obs = arm_obs
        self.arm_last_action = arm_last_action
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # a 480->29 MLP; extra threads only add scheduling stalls on the VM
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(policy_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.reset()

    def reset(self):
        self.last_action = np.zeros(NUM_JOINTS)
        self.phantom_q = DEFAULT_JOINT_POS.copy()     # policy order
        self.phantom_dq = np.zeros(NUM_JOINTS)
        self.terms = [_TermHistory() for _ in range(6)]

    def step(self, q_motor, dq_motor, quat_wxyz, gyro, cmd=(0.0, 0.0, 0.0), arm_override=None):
        """q/dq in motor order (29,). arm_override: {motor index: target} for
        arm joints driven by person_id this step. Returns motor-order targets
        (29,) with the overrides applied."""
        q = np.asarray(q_motor, dtype=float)[JOINT_IDS_MAP]
        dq = np.asarray(dq_motor, dtype=float)[JOINT_IDS_MAP]
        if arm_override and self.arm_obs == "default":
            slots = MOTOR_TO_SLOT[list(arm_override)]
            q[slots] = DEFAULT_JOINT_POS[slots]
            dq[slots] = 0.0
        elif arm_override and self.arm_obs == "phantom":
            slots = MOTOR_TO_SLOT[list(arm_override)]
            q[slots] = self.phantom_q[slots]
            dq[slots] = self.phantom_dq[slots]
        frames = [
            np.asarray(gyro, dtype=float) * ANG_VEL_SCALE,
            projected_gravity(quat_wxyz),
            np.asarray(cmd, dtype=float),
            q - DEFAULT_JOINT_POS,
            dq * JOINT_VEL_SCALE,
            self.last_action,
        ]
        for term, frame in zip(self.terms, frames):
            term.add(frame)
        obs = np.concatenate([t.get() for t in self.terms]).astype(np.float32).reshape(1, -1)
        action = self.session.run(None, {self.input_name: obs})[0].reshape(-1).astype(float)
        if not np.all(np.isfinite(action)):
            raise ValueError("policy returned a non-finite action")
        targets_policy = action * ACTION_SCALE + DEFAULT_JOINT_POS
        # Phantom arms: the arms the policy believes it has follow its own
        # commands (a first-order lag, like a PD-driven arm), so its arm
        # observations and last_action stay consistent with each other.
        new_phantom = self.phantom_q + 0.5 * (targets_policy - self.phantom_q)
        self.phantom_dq = (new_phantom - self.phantom_q) / STEP_DT
        self.phantom_q = new_phantom
        if arm_override:
            for motor, value in arm_override.items():
                slot = MOTOR_TO_SLOT[motor]
                targets_policy[slot] = value
                if self.arm_last_action and self.arm_obs == "real":
                    action[slot] = (value - DEFAULT_JOINT_POS[slot]) / ACTION_SCALE
                elif self.arm_obs == "default":
                    action[slot] = 0.0
                # phantom: the policy keeps its own arm action in last_action
        self.last_action = action
        targets = np.zeros(NUM_JOINTS)
        targets[JOINT_IDS_MAP] = targets_policy
        return targets
