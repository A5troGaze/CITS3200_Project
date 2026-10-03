"""
Synthetic MediaPipe-world-landmark bodies for testing the retargeting pipeline
without a camera. All landmarks are plain {x, y, z, visibility} dicts in
MediaPipe's own world-landmark frame (x right-in-image, y down,
z toward-camera-negative) -- exactly what leader_pose.py exports and what
MimicPipeline.step() consumes.

Only the landmarks the retargeting uses are given real values (shoulders
11/12, elbows 13/14, wrists 15/16, hips 23/24); every other index is a
harmless placeholder. synthetic_poses.py has complete 33-landmark bodies.

Rough body dimensions (metres): hip half-width 0.09, shoulder half-width
0.175, torso (hip-to-shoulder) height 0.5, upper arm 0.28, forearm 0.25.
"""

import math

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

HIP_HALF_WIDTH = 0.09
SHOULDER_HALF_WIDTH = 0.175
TORSO_HEIGHT = 0.5
UPPER_ARM = 0.28
FOREARM = 0.25


def _lm(x, y, z, visibility=1.0):
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def blank_landmarks():
    """33 placeholder landmarks, all at the origin, full visibility."""
    return [_lm(0.0, 0.0, 0.0) for _ in range(33)]


def base_body():
    """Hips + shoulders only, standing upright facing the camera. Callers
    add elbow/wrist positions for whichever arm pose they need."""
    lms = blank_landmarks()
    lms[L_HIP] = _lm(HIP_HALF_WIDTH, 0.0, 0.0)
    lms[R_HIP] = _lm(-HIP_HALF_WIDTH, 0.0, 0.0)
    lms[L_SHOULDER] = _lm(SHOULDER_HALF_WIDTH, -TORSO_HEIGHT, 0.0)
    lms[R_SHOULDER] = _lm(-SHOULDER_HALF_WIDTH, -TORSO_HEIGHT, 0.0)
    return lms


def arms_hanging():
    """Both arms straight down at the sides. Expected: all 8 arm joints
    near 0 rad."""
    lms = base_body()
    for shoulder_idx, elbow_idx, wrist_idx, x in (
        (L_SHOULDER, L_ELBOW, L_WRIST, SHOULDER_HALF_WIDTH),
        (R_SHOULDER, R_ELBOW, R_WRIST, -SHOULDER_HALF_WIDTH),
    ):
        sy = lms[shoulder_idx]["y"]
        lms[elbow_idx] = _lm(x, sy + UPPER_ARM, 0.0)
        lms[wrist_idx] = _lm(x, sy + UPPER_ARM + FOREARM, 0.0)
    return lms


def t_pose():
    """Both arms straight out to the sides. Expected: shoulder_pitch approx
    0, shoulder_roll approx +90 deg (left) / -90 deg (right), elbow approx 0."""
    lms = base_body()
    sy = lms[L_SHOULDER]["y"]
    lms[L_ELBOW] = _lm(SHOULDER_HALF_WIDTH + UPPER_ARM, sy, 0.0)
    lms[L_WRIST] = _lm(SHOULDER_HALF_WIDTH + UPPER_ARM + FOREARM, sy, 0.0)
    lms[R_ELBOW] = _lm(-(SHOULDER_HALF_WIDTH + UPPER_ARM), sy, 0.0)
    lms[R_WRIST] = _lm(-(SHOULDER_HALF_WIDTH + UPPER_ARM + FOREARM), sy, 0.0)
    return lms


def t_pose_elbows_bent_90():
    """T-pose with both elbows bent 90 degrees (forearms point straight up).
    Expected: shoulder angles as in t_pose(), elbow approx +90 deg."""
    lms = t_pose()
    ex_l = lms[L_ELBOW]
    lms[L_WRIST] = _lm(ex_l["x"], ex_l["y"] - FOREARM, ex_l["z"])
    ex_r = lms[R_ELBOW]
    lms[R_WRIST] = _lm(ex_r["x"], ex_r["y"] - FOREARM, ex_r["z"])
    return lms


def arm_straight_forward():
    """Both arms straight out in front (toward the camera). Expected:
    shoulder_pitch approx -90 deg, shoulder_roll approx 0, elbow approx 0."""
    lms = base_body()
    for shoulder_idx, elbow_idx, wrist_idx, x in (
        (L_SHOULDER, L_ELBOW, L_WRIST, SHOULDER_HALF_WIDTH),
        (R_SHOULDER, R_ELBOW, R_WRIST, -SHOULDER_HALF_WIDTH),
    ):
        sy = lms[shoulder_idx]["y"]
        lms[elbow_idx] = _lm(x, sy, -UPPER_ARM)
        lms[wrist_idx] = _lm(x, sy, -(UPPER_ARM + FOREARM))
    return lms


def rotate_about_vertical(landmarks, degrees):
    """Rotate every landmark about MediaPipe's own vertical (y) axis by
    `degrees`, leaving y (height) unchanged. Simulates the same body turned
    to face a different direction relative to the camera, for the
    camera-independence test: torso_frame()/arm_angles() results should be
    unchanged since everything is derived from the body's own geometry."""
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    rotated = []
    for lm in landmarks:
        x, y, z = lm["x"], lm["y"], lm["z"]
        rotated.append(_lm(x * c - z * s, y, x * s + z * c, lm["visibility"]))
    return rotated


def set_visibility(landmarks, index, visibility):
    lms = list(landmarks)
    lm = dict(lms[index])
    lm["visibility"] = visibility
    lms[index] = lm
    return lms
