"""Pinocchio-based arm retargeting: MediaPipe world landmarks -> G1 arm joint angles.

Replaces the geometric estimate in pose_retargeting.py with real inverse
kinematics against the G1's actual skeleton (URDF), so joint angles account
for the robot's real arm segment lengths instead of approximating from
human proportions.

SCOPE: only the same 6 joints pose_retargeting.py already covers (left/right
shoulder pitch, shoulder roll, elbow) - NOT a full 29-DOF solve. This keeps
it a drop-in replacement for retarget_arms_indexed() with no changes needed
to mujoco_pose_controller.py, person_id_live.py, or person_id_replay.py.

======================== BEFORE THIS WILL RUN =========================
Two things below are placeholders because this was written without the
actual G1 URDF in hand. Whoever has Pinocchio + the URDF installed needs to:

  1. Set G1_URDF_PATH to the real file location.
  2. Run this file directly (python3 pinocchio_retargeting.py) - it prints
     every frame name in the model plus a shortlist of likely wrist
     candidates. Find the real left/right wrist frame names from that list
     and fill in LEFT_WRIST_FRAME / RIGHT_WRIST_FRAME below. Common Unitree
     naming looks like "left_wrist_yaw_link" but this MUST be confirmed
     against the real file, not assumed.

See PINOCCHIO_SETUP.md for the full step-by-step.

Everything else (the IK math, joint limit clamping, output format) is
real and should not need changes.
=========================================================================
"""

import numpy as np
import pinocchio as pin

from g1_joint_limits import G1_29DOF_JOINT_LIMITS, clamp_position

# --- VERIFY THESE TWO BEFORE USE -------------------------------------------
G1_URDF_PATH = "/home/ubuntu/CITS3200/Dependencies/g1_description/g1_29dof_rev_1_0.urdf"  # TODO: confirm real path
LEFT_WRIST_FRAME = "left_wrist_yaw_link"    # TODO: verify against the real URDF
RIGHT_WRIST_FRAME = "right_wrist_yaw_link"  # TODO: verify against the real URDF
# -----------------------------------------------------------------------------

# Same motor indices pose_retargeting.py uses (see g1_joint_limits.py / the
# G1JointIndex ordering), so the output dict is a drop-in swap - anything
# downstream that reads {motor_index: angle} doesn't know the difference.
LEFT_SHOULDER_PITCH = 15
LEFT_SHOULDER_ROLL = 16
LEFT_ELBOW = 18
RIGHT_SHOULDER_PITCH = 22
RIGHT_SHOULDER_ROLL = 23
RIGHT_ELBOW = 25

# MediaPipe world-landmark indices (33-point pose model) used as IK targets.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

# Joint names as they appear in the URDF - used to look up joint IDs in the
# Pinocchio model. If these don't match the real URDF, load_model() /
# getJointId() will throw, which is the point (fail loudly, not silently).
ARM_JOINT_NAMES = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"],
}
ARM_OUTPUT_INDICES = {
    "left": [LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL, LEFT_ELBOW],
    "right": [RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL, RIGHT_ELBOW],
}
WRIST_FRAME = {"left": LEFT_WRIST_FRAME, "right": RIGHT_WRIST_FRAME}
SHOULDER_LANDMARK = {"left": L_SHOULDER, "right": R_SHOULDER}
WRIST_LANDMARK = {"left": L_WRIST, "right": R_WRIST}


def load_model(urdf_path=G1_URDF_PATH):
    """Load the G1 model from its URDF file.

    Wrapped in a try/except so a bad path gives a clear, obvious error
    message instead of a confusing Pinocchio internal traceback.
    """
    try:
        model = pin.buildModelFromUrdf(urdf_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load URDF at {urdf_path}. Confirm G1_URDF_PATH at the "
            f"top of this file points to the real file. Original error: {exc}"
        ) from exc
    data = model.createData()
    return model, data


def list_frame_names(model):
    """Print every frame name in the model.

    Run this (via `python3 pinocchio_retargeting.py`) once to find the real
    left/right wrist frame names, then fill in LEFT_WRIST_FRAME /
    RIGHT_WRIST_FRAME near the top of this file. Look under "Likely wrist
    candidates" first - it's just every frame name containing "wrist".
    """
    names = [f.name for f in model.frames]
    wrist_candidates = [n for n in names if "wrist" in n.lower()]
    print("All frame names:")
    for n in names:
        print(f"  {n}")
    print("\nLikely wrist candidates:")
    for n in wrist_candidates:
        print(f"  {n}")
    return names


def _arm_neutral_length(model, data, side):
    """Measure the robot's own upper-arm + forearm length.

    Done via real forward kinematics at the model's neutral (zero) pose,
    rather than hardcoding a number, so this stays correct even if the
    URDF's link lengths differ from what we'd guess. Returns the length
    and the shoulder joint's position in world coordinates (needed below
    to build the IK target point).
    """
    q_zero = pin.neutral(model)
    pin.forwardKinematics(model, data, q_zero)
    pin.updateFramePlacements(model, data)
    shoulder_joint_id = model.getJointId(ARM_JOINT_NAMES[side][0])
    shoulder_pos = data.oMi[shoulder_joint_id].translation
    wrist_frame_id = model.getFrameId(WRIST_FRAME[side])
    wrist_pos = data.oMf[wrist_frame_id].translation
    return float(np.linalg.norm(wrist_pos - shoulder_pos)), shoulder_pos.copy()


def _human_arm_direction(xyz, side):
    """Get the direction the person's arm is pointing (shoulder -> wrist).

    Only the direction is used, not the human's actual arm length - the
    robot's own arm length (from _arm_neutral_length) is what actually
    scales the IK target, since the robot's arms are a different length
    to a human's.
    """
    shoulder = xyz[SHOULDER_LANDMARK[side]]
    wrist = xyz[WRIST_LANDMARK[side]]
    vector = wrist - shoulder
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        raise ValueError(f"Degenerate {side} arm landmarks (shoulder and wrist coincide)")
    return vector / norm


def _solve_arm_ik(model, data, side, target_world_pos, q_full, max_iters=100, tol=1e-4, damping=1e-6):
    """Solve for the 3 arm joint angles that put the wrist at target_world_pos.

    This is damped least-squares IK (a standard Levenberg-Marquardt-style
    approach): each iteration, measure how far the wrist currently is from
    the target, compute the Jacobian (how wrist position changes per unit
    of joint movement), and take a small step in the direction that closes
    the gap. The "damping" term keeps it stable near singular/extended-arm
    poses where a plain least-squares step would blow up.

    Every other joint in the model is held fixed at its current value in
    q_full - we're only solving for this one arm's 3 joints, not the whole
    robot.
    """
    joint_ids = [model.getJointId(name) for name in ARM_JOINT_NAMES[side]]
    velocity_indices = [model.joints[jid].idx_v for jid in joint_ids]
    frame_id = model.getFrameId(WRIST_FRAME[side])
    q = q_full.copy()

    for _ in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pos = data.oMf[frame_id].translation
        error = target_world_pos - current_pos
        if np.linalg.norm(error) < tol:
            break  # close enough, stop early

        # Full Jacobian is for all joints; we only need the 3 columns that
        # correspond to this arm's joints (velocity_indices).
        J_full = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        J = J_full[:3, velocity_indices]  # position rows only (drop rotation rows)

        # Damped least squares step: J^T (J J^T + damping*I)^-1 * error
        JJt = J @ J.T + damping * np.eye(3)
        delta = J.T @ np.linalg.solve(JJt, error)

        for local_idx, vidx in enumerate(velocity_indices):
            q[vidx] += delta[local_idx]

        # Clamp to real hardware limits every iteration (not just at the
        # end) so the solver can't wander into a pose the real robot
        # couldn't physically reach.
        for joint_name, vidx in zip(ARM_JOINT_NAMES[side], velocity_indices):
            q[vidx] = clamp_position(joint_name, q[vidx])

    return q


def retarget_arms_pinocchio(xyz, model, data, q_prev=None):
    """Main entry point - same role as retarget_arms_indexed() in pose_retargeting.py.

    Args:
        xyz: (33, 3) array of MediaPipe world landmarks for one frame.
        model, data: from load_model(), created once at startup and reused.
        q_prev: full-model joint config from the previous frame. Passing
            this in gives the IK solver a starting point close to the last
            solution, so the arm moves smoothly frame-to-frame instead of
            the solver jumping to a different (but equally valid) solution
            each time. Pass None for the very first frame.

    Returns:
        (result, q_full) where result is {motor_index: angle_rad} for the
        same 6 joints pose_retargeting.retarget_arms_indexed() returns
        (drop-in compatible with mujoco_pose_controller.set_targets()), and
        q_full is the full-model config to pass back in as q_prev next frame.
    """
    q_full = q_prev.copy() if q_prev is not None else pin.neutral(model)
    result = {}

    for side in ("left", "right"):
        arm_length, shoulder_pos = _arm_neutral_length(model, data, side)
        direction = _human_arm_direction(xyz, side)
        # Target = robot's own shoulder position + (human arm direction *
        # robot's own arm length). This is the point in the world we want
        # the robot's wrist to reach.
        target_pos = shoulder_pos + direction * arm_length

        q_full = _solve_arm_ik(model, data, side, target_pos, q_full)

        # Pull the 3 solved joint angles for this arm out of the full
        # config and map them to the motor indices the rest of the
        # pipeline expects.
        for joint_name, output_index in zip(ARM_JOINT_NAMES[side], ARM_OUTPUT_INDICES[side]):
            vidx = model.joints[model.getJointId(joint_name)].idx_v
            result[output_index] = float(q_full[vidx])

    return result, q_full


if __name__ == "__main__":
    # Quick standalone check - run this file directly to load the model and
    # print its frame names. Use this to find the real wrist frame names
    # (see the top of the file / PINOCCHIO_SETUP.md step 3).
    model, data = load_model()
    print(f"Loaded model with {model.nq} DOF, {len(model.frames)} frames.\n")
    list_frame_names(model)
