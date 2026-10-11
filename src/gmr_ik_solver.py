'''
gmr_ik_solver.py

Wrapper around GMR (General Motion Retargeting). Takes the human pose from
the camera, solves inverse kinematics for the G1 one frame at a time, and
returns the arm/waist angles as {joint name: angle in rad}, clamped to the
real joint limits.

Uses our own IK config (data/mediapipe_to_g1.json), registered into GMR at
runtime, so GMR's own files are never modified.

This file used to be called: gmr_retarget.pt
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import contextlib
import io
import os
import time

import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS


#----------------------------------------------------------
# Constants
#----------------------------------------------------------
# our IK config, next to gestures.json in the data folder
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mediapipe_to_g1.json")
SRC_HUMAN = "mediapipe"     # human data format name used by the config
TGT_ROBOT = "unitree_g1"    # GMR's built-in G1 model

# Motor index -> joint name, built from the limits table (which is keyed
# by name and carries the LowCmd index).
MOTOR_JOINT_NAMES = [None] * 29
for _name, _lim in G1_29DOF_JOINT_LIMITS.items():
    MOTOR_JOINT_NAMES[_lim.index] = _name
assert None not in MOTOR_JOINT_NAMES    # every one of the 29 motors must be named

# joint groups, picked out by name
LEG_JOINTS = [n for n in MOTOR_JOINT_NAMES if "hip" in n or "knee" in n or "ankle" in n]
ARM_JOINTS = [n for n in MOTOR_JOINT_NAMES if any(k in n for k in ("shoulder", "elbow", "wrist"))]
WAIST_JOINTS = {
    "3dof": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "yaw": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "off": [],
}


#----------------------------------------------------------
# Helpers
#----------------------------------------------------------
def commanded_joint_names(waist="off", legs=False):
    '''Joints we send targets for. With waist="yaw", roll/pitch are
    still commanded but always at 0 (a locked waist ignores them).'''
    names = list(ARM_JOINTS) + WAIST_JOINTS[waist]
    if legs:
        names += LEG_JOINTS
    return names


def clamp_to_limits(name, value):
    '''Clamp an angle (rad) into the joint's real position range.'''
    lim = G1_29DOF_JOINT_LIMITS[name]
    return min(max(float(value), lim.lower), lim.upper)


def _register_config():
    '''Add our IK config to GMR's list of configs, so GMR can find it by name.'''
    # imported here (not at the top) so GMR only loads when it's needed
    from general_motion_retargeting.params import IK_CONFIG_DICT

    IK_CONFIG_DICT.setdefault(SRC_HUMAN, {})[TGT_ROBOT] = CONFIG_PATH


#----------------------------------------------------------
# Retargeter
#----------------------------------------------------------
class GmrIKSolver:      # Renamed from 'GmrRetargeter'
    '''One GMR instance, solved per frame.'''

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
        #   (pose_to_target builds targets directly in G1 link frames), so
        #   GMR's per-body scipy scale/offset passes return their input;
        # * mink's solve_ik re-validates the configuration against the joint
        #   limits on every IK step by building a new ConfigurationLimit
        #   (~0.6 ms each). The solution is clamped to the real limits below.
        self.gmr.scale_human_data = lambda data, root, table: data
        self.gmr.offset_human_data = lambda data, pos_offsets, rot_offsets: data
        self.gmr.configuration.check_limits = lambda *args, **kwargs: None

        # joint name -> position in the model's qpos array
        self.qpos_adr = {
            name: int(self.model.joint(name).qposadr[0]) for name in MOTOR_JOINT_NAMES
        }
        # original orientation weights of each body, so they can be rescaled per frame
        self.base_rot_cost = {
            body: np.array(task.orientation_cost, dtype=float).copy()
            for body, task in self.gmr.human_body_to_task1.items()
        }

        # solve settings: stop once the time budget, call cap or convergence is hit
        self.budget_ms = budget_ms
        self.max_calls = max_calls
        self.converge_tol = converge_tol

        # stats from the last solve
        self.last_solve_ms = 0.0
        self.last_solve_cpu_ms = 0.0
        self.last_calls = 0
        self.last_qpos = None

    #------------------------------------------------------
    # Solve
    #------------------------------------------------------
    def solve(self, human_data, rot_weight_scale=None):
        '''Retarget one frame. Returns {joint_name: angle_rad} for all 29
        joints, clamped to the real G1 limits.

        One GMR retarget() call stops after at most 11 IK steps, or as soon
        as an IK step improves the error by less than 0.001. After a large
        jump (e.g. arms down -> overhead) that can leave the solution short
        of the target. retarget() is warm-started, so it is simply called
        again until the error stops improving, capped by budget_ms and
        max_calls. A slowly moving person converges in one or two calls.
        '''
        # optionally rescale how much each body's orientation matters this frame
        if rot_weight_scale:
            for body, task in self.gmr.human_body_to_task1.items():
                task.set_orientation_cost(self.base_rot_cost[body] * rot_weight_scale.get(body, 1.0))

        t0 = time.perf_counter()    # wall-clock timer
        c0 = time.thread_time()     # CPU time of this thread
        prev_err = np.inf
        calls = 0

        # keep calling retarget() until the error stops improving or we run out of budget
        while True:
            qpos = self.gmr.retarget(human_data)
            calls += 1
            err = self.gmr.error1()
            elapsed = (time.perf_counter() - t0) * 1000.0
            if prev_err - err < self.converge_tol or calls >= self.max_calls or elapsed >= self.budget_ms:
                break
            prev_err = err

        # record stats for this solve
        self.last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self.last_solve_cpu_ms = (time.thread_time() - c0) * 1000.0
        self.last_calls = calls
        self.last_error = float(err)

        # never let a bad solve poison the warm start or reach the robot
        if not np.all(np.isfinite(qpos)):
            self.reset(self.last_qpos)
            raise ValueError("GMR returned a non-finite solution")

        self.last_qpos = qpos
        # joint name -> angle, clamped to the real limits
        return {name: clamp_to_limits(name, qpos[adr]) for name, adr in self.qpos_adr.items()}

    def reset(self, qpos=None):
        '''Reset the warm start (to qpos, or the model's rest pose).'''
        q = self.model.qpos0.copy() if qpos is None else np.asarray(qpos, dtype=float)
        self.gmr.configuration.update(q)