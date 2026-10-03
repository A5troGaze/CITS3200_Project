"""
GMR (General Motion Retargeting) wrapper for person_id.

Loads GMR's unitree_g1 model with person_id's own IK config
(configs/mediapipe_to_g1.json, registered into GMR's IK_CONFIG_DICT at
runtime so GMR's files stay untouched), solves one frame at a time
(warm-started: GMR keeps its mink configuration between calls) and maps
the solution to G1 motor indices by joint name, clamped to the real joint
limits in g1_joint_limits.py.
"""

import contextlib
import io
import os
import time

import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "mediapipe_to_g1.json")
SRC_HUMAN = "mediapipe"
TGT_ROBOT = "unitree_g1"

# Motor index -> joint name, derived from the limits table (which is keyed
# by name and carries the LowCmd index). Checked against both MJCFs in
# tests/test_g1_mapping.py.
MOTOR_JOINT_NAMES = [None] * 29
for _name, _lim in G1_29DOF_JOINT_LIMITS.items():
    MOTOR_JOINT_NAMES[_lim.index] = _name
assert None not in MOTOR_JOINT_NAMES

LEG_JOINTS = [n for n in MOTOR_JOINT_NAMES if "hip" in n or "knee" in n or "ankle" in n]
ARM_JOINTS = [n for n in MOTOR_JOINT_NAMES if any(k in n for k in ("shoulder", "elbow", "wrist"))]
WAIST_JOINTS = {
    "3dof": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "yaw": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "off": [],
}


def commanded_joint_names(waist="off", legs=False):
    """Joints person_id sends targets for. With waist="yaw", roll/pitch are
    still commanded but always at 0 (a locked waist ignores them)."""
    names = list(ARM_JOINTS) + WAIST_JOINTS[waist]
    if legs:
        names += LEG_JOINTS
    return names


def clamp_to_limits(name, value):
    lim = G1_29DOF_JOINT_LIMITS[name]
    return min(max(float(value), lim.lower), lim.upper)


def _register_config():
    from general_motion_retargeting.params import IK_CONFIG_DICT

    IK_CONFIG_DICT.setdefault(SRC_HUMAN, {})[TGT_ROBOT] = CONFIG_PATH


class GmrRetargeter:
    """One GMR instance, solved per frame."""

    def __init__(self, solver="daqp", damping=0.5, max_iter=10, quiet=True,
                 budget_ms=12.0, max_calls=40, converge_tol=1e-3):
        # GMR prints its whole model on construction; keep the console clean.
        sink = io.StringIO() if quiet else None
        with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
            from general_motion_retargeting import GeneralMotionRetargeting

            _register_config()
            self.gmr = GeneralMotionRetargeting(
                src_human=SRC_HUMAN, tgt_robot=TGT_ROBOT, solver=solver,
                damping=damping, verbose=False,
            )
        self.gmr.max_iter = max_iter
        self.model = self.gmr.model
        # Per-instance speed-ups (GMR's files are not modified):
        # * our config has every scale 1.0 and every offset identity
        #   (mediapipe_to_gmr builds targets directly in G1 link frames), so
        #   GMR's per-body scipy scale/offset passes return their input;
        # * mink's solve_ik re-validates the configuration against the joint
        #   limits on every IK step by building a new ConfigurationLimit
        #   (~0.6 ms each). The solution is clamped to the real limits below.
        self.gmr.scale_human_data = lambda data, root, table: data
        self.gmr.offset_human_data = lambda data, pos_offsets, rot_offsets: data
        self.gmr.configuration.check_limits = lambda *args, **kwargs: None
        self.qpos_adr = {
            name: int(self.model.joint(name).qposadr[0]) for name in MOTOR_JOINT_NAMES
        }
        self.base_rot_cost = {
            body: np.array(task.orientation_cost, dtype=float).copy()
            for body, task in self.gmr.human_body_to_task1.items()
        }
        self.budget_ms = budget_ms
        self.max_calls = max_calls
        self.converge_tol = converge_tol
        self.last_solve_ms = 0.0
        self.last_solve_cpu_ms = 0.0
        self.last_calls = 0
        self.last_qpos = None

    def solve(self, human_data, rot_weight_scale=None):
        """Retarget one frame. Returns {joint_name: angle_rad} for all 29
        joints, clamped to the real G1 limits.

        One GMR retarget() call stops after at most 11 IK steps, or as soon
        as an IK step improves the error by less than 0.001. After a large
        jump (e.g. arms down -> overhead) that can leave the solution short
        of the target. retarget() is warm-started, so it is simply called
        again until the error stops improving, capped by budget_ms and
        max_calls. A slowly moving person converges in one or two calls.
        """
        if rot_weight_scale:
            for body, task in self.gmr.human_body_to_task1.items():
                task.set_orientation_cost(self.base_rot_cost[body] * rot_weight_scale.get(body, 1.0))
        t0 = time.perf_counter()
        c0 = time.thread_time()
        prev_err = np.inf
        calls = 0
        while True:
            qpos = self.gmr.retarget(human_data)
            calls += 1
            err = self.gmr.error1()
            elapsed = (time.perf_counter() - t0) * 1000.0
            if prev_err - err < self.converge_tol or calls >= self.max_calls or elapsed >= self.budget_ms:
                break
            prev_err = err
        self.last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self.last_solve_cpu_ms = (time.thread_time() - c0) * 1000.0
        self.last_calls = calls
        self.last_error = float(err)
        if not np.all(np.isfinite(qpos)):
            # Never let a bad solve poison the warm start or reach the robot.
            self.reset(self.last_qpos)
            raise ValueError("GMR returned a non-finite solution")
        self.last_qpos = qpos
        return {name: clamp_to_limits(name, qpos[adr]) for name, adr in self.qpos_adr.items()}

    def reset(self, qpos=None):
        """Reset the warm start (to qpos, or the model's rest pose)."""
        q = self.model.qpos0.copy() if qpos is None else np.asarray(qpos, dtype=float)
        self.gmr.configuration.update(q)
