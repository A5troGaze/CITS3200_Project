"""Convert 33 MediaPipe world landmarks into six G1 arm-joint targets.

The conversion estimates shoulder pitch, shoulder roll, and elbow flexion for
both arms relative to the person's torso. It does not control shoulder yaw,
wrists, waist, or legs, and it is not a full inverse-kinematics solver.
"""

import math
import numpy as np

# G1 29-DOF motor indices. Kept here so the maths-only dry run does not
# require unitree_sdk2py or an active CycloneDDS installation.
LEFT_SHOULDER_PITCH = 15
LEFT_SHOULDER_ROLL = 16
LEFT_ELBOW = 18
RIGHT_SHOULDER_PITCH = 22
RIGHT_SHOULDER_ROLL = 23
RIGHT_ELBOW = 25

# MediaPipe indices used to construct the torso and both arm chains.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24


def _validated_xyz(values):
    # Detect incorrect shape, non-numeric values, NaN, and infinity early.
    try:
        xyz = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmark coordinates must be numeric") from exc
    if xyz.shape != (33, 3):
        raise ValueError(f"Expected landmark shape (33, 3), got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("Landmarks contain NaN or infinite coordinates")
    return xyz


def _unit(vector, label):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        # Coincident landmarks cannot define a reliable limb direction.
        raise ValueError(f"Degenerate {label} landmarks")
    return vector / norm


def _torso_frame(xyz):
    # Build person-relative right/up/forward axes, independent of camera facing.
    shoulder_mid = (xyz[L_SHOULDER] + xyz[R_SHOULDER]) / 2.0
    hip_mid = (xyz[L_HIP] + xyz[R_HIP]) / 2.0
    right = _unit(xyz[R_SHOULDER] - xyz[L_SHOULDER], "shoulder")
    up_seed = _unit(shoulder_mid - hip_mid, "torso")
    forward = _unit(np.cross(right, up_seed), "torso-frame")
    up = _unit(np.cross(forward, right), "torso-frame")  # Re-orthogonalize.
    return right, up, forward


def _arm_angles(xyz, shoulder, elbow, wrist, right, up, forward):
    upper = _unit(xyz[elbow] - xyz[shoulder], "upper-arm")
    forearm = _unit(xyz[wrist] - xyz[elbow], "forearm")
    lateral = float(np.dot(upper, right))
    vertical_down = -float(np.dot(upper, up))
    forward_amount = float(np.dot(upper, forward))

    # The non-negative denominator avoids atan2(0, -0) returning pi in a T-pose.
    pitch = math.atan2(forward_amount, math.hypot(lateral, vertical_down))
    # G1 convention: outward is positive on the left and negative on the right.
    outward = -lateral
    roll = math.atan2(outward, vertical_down)
    elbow_flex = math.acos(float(np.clip(np.dot(upper, forearm), -1.0, 1.0)))
    return pitch, roll, elbow_flex


def retarget_arms_indexed(values):
    xyz = _validated_xyz(values)
    right, up, forward = _torso_frame(xyz)
    lp, lr, le = _arm_angles(
        xyz, L_SHOULDER, L_ELBOW, L_WRIST, right, up, forward
    )
    rp, rr, re = _arm_angles(
        xyz, R_SHOULDER, R_ELBOW, R_WRIST, right, up, forward
    )
    return {
        LEFT_SHOULDER_PITCH: lp,
        LEFT_SHOULDER_ROLL: lr,
        LEFT_ELBOW: le,
        RIGHT_SHOULDER_PITCH: rp,
        RIGHT_SHOULDER_ROLL: rr,
        RIGHT_ELBOW: re,
    }


def json_landmarks_to_xyz(landmarks):
    # Convert exported JSON dictionaries containing x/y/z fields.
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        raise ValueError("Expected a list containing 33 world landmarks")
    try:
        return _validated_xyz([[item["x"], item["y"], item["z"]] for item in landmarks])
    except (KeyError, TypeError) as exc:
        raise ValueError("Every landmark must contain numeric x, y and z fields") from exc


def mediapipe_landmarks_to_xyz(landmarks):
    # Convert PoseLandmarker objects containing .x/.y/.z attributes.
    if landmarks is None or len(landmarks) != 33:
        raise ValueError("Expected 33 MediaPipe world landmarks")
    try:
        return _validated_xyz([[item.x, item.y, item.z] for item in landmarks])
    except AttributeError as exc:
        raise ValueError("Every MediaPipe landmark must contain x, y and z") from exc
