"""
Skeleton for: MediaPipe pose landmarks -> G1 joint targets -> MuJoCo viz.

This is NOT a working retargeter. The IK/retargeting step (marked TODO
below) is the actual unsolved piece of work - see the note at the bottom
of this file for why, and what filling it in requires. Everything else
here (reading your team's landmark JSON, calling into GMR's own MuJoCo
viewer, checking output against g1_joint_limits.py) is real and runnable
once GMR and its dependencies are on the machine per SETUP.md.

Layout assumed:
    g1_joint_limits.py                 <- from Kinematics_Reference/
    pose_to_g1_pipeline_skeleton.py    <- this file
    (GMR installed as a package: `pip3 install -e .` per SETUP.md)
"""

import json

import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS, validate_command

# GMR ships a ready-made MuJoCo viewer for exactly this: load the G1 MJCF,
# and each frame set data.qpos = [root_pos(3), root_quat(4), dof_pos(29)]
# then mj_forward() + render. See general_motion_retargeting/robot_motion_viewer.py
# and scripts/vis_robot_motion.py / scripts/xsens_live_streaming.py in the
# GMR repo for the pattern this is modeled on.
from general_motion_retargeting import RobotMotionViewer


# --- MediaPipe Pose Landmarker indices used below (0-32 standard schema) ---
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# G1 DOF order used by the MJCF / RobotMotionViewer.step()'s dof_pos array -
# must match g1_joint_limits.G1_29DOF_JOINT_LIMITS insertion order.
G1_JOINT_ORDER = list(G1_29DOF_JOINT_LIMITS.keys())


def load_leader_landmarks(export_json_path):
    """Read the JSON your team's pose_test-world_landmarks.py writes:
    a list of {frame_id, timestamp_ms, leader_id, bbox, landmarks_world_m[33]}.
    landmarks_world_m[i] is {x, y, z} in metres, MediaPipe's world-landmark
    frame (roughly hip-centred), one entry per frame.
    """
    with open(export_json_path) as f:
        frames = json.load(f)
    return frames


def landmarks_to_xyz(landmarks_world_m):
    """[{x,y,z}, ...] (len 33) -> (33, 3) numpy array."""
    return np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks_world_m])


def estimate_human_targets(xyz):
    """Turn the 33 raw points into the handful of named 3D targets a
    retargeter needs (pelvis, hips, knees, feet, shoulders, elbows,
    wrists - the same body names GMR's ik_configs/*_to_g1.json files key
    on, e.g. smplx_to_g1.json's ik_match_table1).

    MediaPipe has no "pelvis" landmark, so it's approximated as the
    hip midpoint here - same idea for anything else not directly present.
    This only produces POSITIONS. It does NOT produce the per-joint
    ORIENTATIONS (quaternions) that GMR's ik_match_table entries also
    carry - see the TODO note below for why that matters.
    """
    return {
        "pelvis": (xyz[L_HIP] + xyz[R_HIP]) / 2,
        "left_hip": xyz[L_HIP],
        "right_hip": xyz[R_HIP],
        "left_knee": xyz[L_KNEE],
        "right_knee": xyz[R_KNEE],
        "left_foot": xyz[L_ANKLE],
        "right_foot": xyz[R_ANKLE],
        "left_shoulder": xyz[L_SHOULDER],
        "right_shoulder": xyz[R_SHOULDER],
        "left_elbow": xyz[L_ELBOW],
        "right_elbow": xyz[R_ELBOW],
        "left_wrist": xyz[L_WRIST],
        "right_wrist": xyz[R_WRIST],
    }


def retarget_to_g1(human_targets):
    """TODO: this is the real kinematics-subgroup work, not a stub to fill
    in casually. Two honest options:

    1. Adapt GMR's own IK (general_motion_retargeting.motion_retarget.
       GeneralMotionRetargeting) by writing a new ik_config (see
       ik_configs/smplx_to_g1.json for the schema: robot link name ->
       [human joint name, position weight, rotation weight, pos offset,
       quat offset]) that targets ONLY position for each entry (drop/zero
       the rotation weight) since raw MediaPipe landmarks carry no limb
       orientation - only 3D points. Positions alone under-constrain some
       joints (e.g. forearm/shin roll about their own long axis), so
       expect degraded accuracy on those DOF versus GMR's SMPL-X pipeline,
       which starts from a fitted body mesh with real bone orientations.
    2. Skip GMR's IK and solve it yourself with Pinocchio (already in
       SETUP.md's dependency list) - build a small analytic/numeric IK
       for the leg and arm chains against human_targets, respecting
       G1_29DOF_JOINT_LIMITS as the joint bounds in the solve.

    Either way, the OUTPUT contract this pipeline expects is a dict
    {joint_name: angle_rad} covering all 29 names in G1_JOINT_ORDER -
    that's what wires cleanly into validate_command() and
    RobotMotionViewer.step() below.
    """
    raise NotImplementedError(
        "retargeting/IK not implemented - see the TODO note above"
    )


def run(export_json_path):
    frames = load_leader_landmarks(export_json_path)
    viewer = RobotMotionViewer(robot_type="unitree_g1", motion_fps=30)

    try:
        for frame in frames:
            xyz = landmarks_to_xyz(frame["landmarks_world_m"])
            human_targets = estimate_human_targets(xyz)

            joint_angles = retarget_to_g1(human_targets)  # {name: rad}, 29 entries

            violations = validate_command(joint_angles)
            if violations:
                # Don't silently clamp and move on during dev - a limit hit
                # usually means the retargeting step, not the robot, is wrong.
                print(f"frame {frame['frame_id']}: {violations}")
                continue

            dof_pos = np.array([joint_angles[name] for name in G1_JOINT_ORDER])
            root_pos = human_targets["pelvis"]
            root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity; refine if you track root orientation

            viewer.step(root_pos=root_pos, root_rot=root_quat, dof_pos=dof_pos)
    finally:
        viewer.close()


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
