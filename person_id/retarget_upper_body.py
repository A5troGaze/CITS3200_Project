"""
Analytic (non-IK) retargeting from a leader's MediaPipe world landmarks to
G1 upper-body joint targets.

Only the 14 arm joints (G1_29DOF_JOINT_LIMITS indices 15-28) are retargeted.
Waist and legs (indices 0-14) are held at the robot's current LowState
value (or zero in --dry-run with no LowState yet) — this module never
attempts locomotion or balance.

Same-side mapping: the person's LEFT arm drives the robot's LEFT arm (not a
mirror). To flip this, swap the "left"/"right" landmark index pairs passed
into retarget_frame()/Retargeter.step() — that is the one line to change.

--- Coordinate frames --------------------------------------------------
MediaPipe world landmarks: x right-in-image, y down, z toward-the-camera-
is-negative (see mediapipe_to_robot for the exact mapping this assumes).
G1 base frame (and this module's "robot frame" from here on): x forward,
y left, z up — this matches the real URDF (confirmed below), not a guess:
waist_yaw_joint's axis is (0,0,1) chained through zero-origin joints
straight off `pelvis`, waist_roll_joint's axis is (1,0,0), waist_pitch's is
(0,1,0) — i.e. roll-about-x, pitch-about-y, yaw-about-z, the standard
REP-103-style x-forward/y-left/z-up body convention.
(source: unitree_ros g1_29dof_rev_1_0.urdf, <joint name="waist_*_joint">)

--- Why only pitch, roll and elbow are exactly solvable, and yaw is not ---
Per the same URDF, shoulder_pitch axis is (0,1,0) [torso lateral/y],
shoulder_roll axis is (1,0,0) [chained off the pitch link], and
shoulder_yaw axis is (0,0,1) *in the shoulder_roll_link's own frame* — and
shoulder_roll_link -> shoulder_yaw_link's origin translation is almost
entirely along that same local z (0, 0.00624, -0.1032 for the left arm),
i.e. shoulder_yaw's axis is (to within the small mounting offset) the
upper arm's own long axis. Rotating a segment about its own long axis
doesn't change which way that axis points, so shoulder_yaw cannot be
recovered from the upper-arm DIRECTION vector at all — only pitch and roll
can (a direction vector has exactly 2 degrees of freedom, matching pitch+
roll exactly). Yaw is instead estimated from the forearm's bend plane
around that axis (see arm_angles), which is why it is flagged as the
weakest-observable angle here and conservatively clamped.

The ~16 degree mounting tilts in shoulder_pitch_joint's and
shoulder_roll_joint's <origin rpy="..."> (+0.27931 / -0.27925 rad) very
nearly cancel (residual ~0.00006 rad) — small enough against MediaPipe's
own noise floor that this module treats the chain as clean sequential
rotations about the torso's own axes (Ry(pitch) then Rx(roll)) rather than
modelling the anatomical mounting tilt. Documented here rather than
silently assumed.

Sources checked line-by-line against unitreerobotics/unitree_ros
robots/g1_description/g1_29dof_rev_1_0.urdf (raw file fetched directly,
not from memory) for every joint named below.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS, clamp_position, validate_command

# --- MediaPipe Pose Landmarker indices used below (0-32 standard schema) ---
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

_SIDE_LANDMARKS = {
    "left": {"shoulder": L_SHOULDER, "elbow": L_ELBOW, "wrist": L_WRIST},
    "right": {"shoulder": R_SHOULDER, "elbow": R_ELBOW, "wrist": R_WRIST},
}

# The 4 solved joints per side, in the order arm_angles() returns them.
_SIDE_ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint"],
}
_SIDE_WRIST_JOINTS = {
    "left": ["left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "right": ["right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
}

G1_JOINT_ORDER = list(G1_29DOF_JOINT_LIMITS.keys())

# (R2) One documented rotation matrix, robot = R @ mediapipe. Never do axis
# swaps anywhere else in this pipeline.
_MP_TO_ROBOT = np.array([
    [0.0, 0.0, -1.0],   # robot forward (x) = -mediapipe z (toward-camera is forward)
    [1.0, 0.0, 0.0],    # robot left (y)    =  mediapipe x (image-right is the person's left)
    [0.0, -1.0, 0.0],   # robot up (z)      = -mediapipe y (image-down is down)
])


def mediapipe_to_robot(points):
    """(R2) MediaPipe world-landmark frame -> G1 base frame.

    points: array-like (3,) or (N,3) of MediaPipe world landmarks (x right
    in image, y down, z toward-camera-negative).
    Returns the same shape in the G1 base frame (x forward, y left, z up).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        return _MP_TO_ROBOT @ pts
    return pts @ _MP_TO_ROBOT.T


def landmarks_to_xyz(landmarks_world_m):
    """[{x,y,z,...}, ...] (len 33, dicts or MediaPipe landmark objects) -> (33,3) ndarray."""
    def _xyz(lm):
        if isinstance(lm, dict):
            return lm["x"], lm["y"], lm["z"]
        return lm.x, lm.y, lm.z
    return np.array([_xyz(lm) for lm in landmarks_world_m], dtype=float)


def _visibility(lm):
    if isinstance(lm, dict):
        return lm.get("visibility", 1.0)
    return getattr(lm, "visibility", 1.0)


def landmark_ok(landmarks_world_m, side, threshold=0.5):
    """(R7) True if shoulder, elbow and wrist on `side` are all at/above
    `threshold` visibility. Checked on the RAW (pre-rotation) landmarks
    since visibility isn't a coordinate."""
    idx = _SIDE_LANDMARKS[side]
    return all(
        _visibility(landmarks_world_m[i]) >= threshold
        for i in (idx["shoulder"], idx["elbow"], idx["wrist"])
    )


def torso_frame(xyz_robot):
    """(R4) Build a torso-attached frame from robot-frame landmarks.

    origin: shoulder midpoint.
    Returns (origin, R_torso) where R_torso's rows are the unit
    (forward, lateral, up) axes expressed in the robot base frame, so
    `R_torso @ (world_vec)` expresses a world-frame vector (e.g. an
    elbow-shoulder direction) in this torso-attached frame. Identical
    result regardless of which way the torso is turned relative to the
    camera, since every input comes from the body itself.
    """
    l_sh, r_sh = xyz_robot[L_SHOULDER], xyz_robot[R_SHOULDER]
    l_hip, r_hip = xyz_robot[L_HIP], xyz_robot[R_HIP]

    origin = (l_sh + r_sh) / 2.0

    lateral = l_sh - r_sh
    lateral = lateral / np.linalg.norm(lateral)

    hip_mid = (l_hip + r_hip) / 2.0
    up_raw = origin - hip_mid
    # Gram-Schmidt against lateral so a forward-leaning torso doesn't leak
    # into the lateral axis.
    up = up_raw - np.dot(up_raw, lateral) * lateral
    up = up / np.linalg.norm(up)

    forward = np.cross(lateral, up)
    forward = forward / np.linalg.norm(forward)

    R_torso = np.stack([forward, lateral, up], axis=0)
    return origin, R_torso


def arm_angles(upper, fore, side, prev_yaw=0.0, yaw_clamp=math.radians(60)):
    """(R1) upper, fore: (elbow-shoulder), (wrist-elbow) direction vectors,
    already expressed in the torso frame (torso_frame() @ vector). side:
    "left" or "right" (kept for symmetry with the rest of the module; the
    formulas below are identical for both sides — see the docstring above
    for why the URDF's mirrored shoulder_roll limits do not require an
    extra sign flip here).

    Returns (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow), radians.
    """
    del side  # formulas are side-symmetric; kept in the signature for clarity/future use

    upper_hat = upper / np.linalg.norm(upper)
    uf, ul, uu = upper_hat  # forward, lateral, up components

    # Shoulder pitch: derived from shoulder_pitch's axis (0,1,0 = lateral)
    # and the URDF's arm-hanging reference direction (0,0,-1). Verified
    # against the URDF: applying a positive rotation about +lateral to the
    # hanging vector swings the arm BACKWARD (-forward), so raising the arm
    # forward is a NEGATIVE angle — confirmed by the sanity target "arm
    # straight forward -> shoulder_pitch approx -90 deg".
    # Valid for |shoulder_pitch| <= 90 deg (covers hanging/forward/T-pose/
    # elbow-bend sanity checks and ordinary mirroring); beyond that the
    # cos(pitch) >= 0 branch below can't distinguish the sign, and the
    # hardware clamp downstream is the safety net for that edge case.
    cos_pitch = math.sqrt(max(ul * ul + uu * uu, 1e-12))
    shoulder_pitch = math.atan2(-uf, cos_pitch)

    # Shoulder roll: derived the same way from shoulder_roll's axis
    # (1,0,0 = forward, in the once-pitched frame). T-pose (lateral=+/-1,
    # up=0) gives +/-90 deg, matching the URDF's mirrored left/right
    # shoulder_roll limit ranges without needing a side-based sign flip
    # here (right arm's T-pose lateral component is already negative in
    # this shared torso frame).
    # Gimbal lock at |shoulder_pitch| ~= 90 deg (arm pointing straight
    # forward/back): lateral and up both go to ~0, so atan2 on two
    # near-zero, noise-dominated values returns an arbitrary angle instead
    # of the well-defined 0 the "arm straight forward" sanity check
    # expects. cos_pitch (already computed above) is exactly the
    # magnitude of (lateral, up), so it is the natural guard.
    if cos_pitch < 1e-3:
        shoulder_roll = 0.0
    else:
        shoulder_roll = math.atan2(ul, -uu)

    # Elbow: angle between the upper-arm and forearm direction vectors.
    # 0 when the arm is straight (upper and forearm point the same way),
    # growing positive as the elbow bends — equivalent to "pi minus the
    # angle between the shoulder-side and wrist-side segments meeting at
    # the elbow vertex". Always >= 0: this never invokes the elbow joint's
    # small (-60 deg) hyperextension allowance, since neither MediaPipe nor
    # human anatomy bends an elbow backward past straight.
    fore_hat = fore / np.linalg.norm(fore)
    elbow = math.acos(np.clip(np.dot(upper_hat, fore_hat), -1.0, 1.0))

    # Shoulder yaw (weakest observable, per the module docstring): the
    # component of the forearm direction perpendicular to the upper-arm
    # axis, i.e. its rotation angle "around" the upper arm. Undefined when
    # the elbow is too straight to establish a bend plane (fore_perp near
    # zero) — hold the previous raw yaw estimate rather than compute a
    # noisy/undefined angle in that case. Clamped conservatively to +/-60
    # deg (well inside the joint's real +/-150 deg range) because this
    # derivation, unlike pitch/roll/elbow above, could not be verified
    # against a real orientation signal (MediaPipe gives no forearm
    # orientation, only a position) — flagged rather than trusted fully.
    fore_perp = fore_hat - np.dot(fore_hat, upper_hat) * upper_hat
    fore_perp_norm = np.linalg.norm(fore_perp)
    if fore_perp_norm < 1e-3:
        shoulder_yaw = prev_yaw
    else:
        fore_perp_hat = fore_perp / fore_perp_norm
        # Reference "zero yaw" direction: torso "up" projected into the
        # plane perpendicular to the upper arm.
        up_world = np.array([0.0, 0.0, 1.0])
        ref_perp = up_world - np.dot(up_world, upper_hat) * upper_hat
        ref_norm = np.linalg.norm(ref_perp)
        if ref_norm < 1e-3:
            # Upper arm is ~parallel to torso "up" (arm raised straight
            # overhead) -- fall back to "forward" as the reference axis.
            fwd_world = np.array([1.0, 0.0, 0.0])
            ref_perp = fwd_world - np.dot(fwd_world, upper_hat) * upper_hat
            ref_norm = np.linalg.norm(ref_perp)
        ref_perp_hat = ref_perp / ref_norm

        sin_part = np.dot(np.cross(ref_perp_hat, fore_perp_hat), upper_hat)
        cos_part = np.dot(ref_perp_hat, fore_perp_hat)
        shoulder_yaw = math.atan2(sin_part, cos_part)
        shoulder_yaw = max(-yaw_clamp, min(yaw_clamp, shoulder_yaw))

    return shoulder_pitch, shoulder_roll, shoulder_yaw, elbow


class TargetFilter:
    """(R5) Per-joint exponential smoothing, then a per-tick rate limit
    from the joint's real velocity limit and dt, then clamp_position, then
    validate_command as the last gate. A tick that still fails validation
    (should be unreachable since clamp_position already enforces the same
    bounds validate_command checks, but this is the explicit last-line gate
    the spec calls for) keeps the previous value and logs the joint.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed = {}
        self.last_sent = {}

    def step(self, raw_targets, dt):
        """raw_targets: {joint_name: rad} for the joints updated this
        call. Returns {joint_name: rad}, same keys, filtered."""
        out = {}
        for name, raw in raw_targets.items():
            prev_smoothed = self.smoothed.get(name)
            smoothed = raw if prev_smoothed is None else (
                self.alpha * raw + (1 - self.alpha) * prev_smoothed
            )
            self.smoothed[name] = smoothed

            prev_sent = self.last_sent.get(name)
            if prev_sent is None:
                rate_limited = smoothed
            else:
                max_step = G1_29DOF_JOINT_LIMITS[name].velocity * dt
                delta = max(-max_step, min(max_step, smoothed - prev_sent))
                rate_limited = prev_sent + delta

            clamped = clamp_position(name, rate_limited)

            violations = validate_command({name: clamped})
            if violations:
                print(f"retarget_upper_body: rejecting {name}: {violations}")
                clamped = prev_sent if prev_sent is not None else clamped

            self.last_sent[name] = clamped
            out[name] = clamped
        return out


@dataclass
class RetargetResult:
    """(R6) Output contract used by mujoco_link.py and the tests."""
    q: dict = field(default_factory=dict)          # all 29 joint_name -> rad
    q_array: "np.ndarray" = None                    # (29,) in G1JointIndex order
    updated_sides: set = field(default_factory=set)  # subset of {"left", "right"}
    frame_id: int = 0


class Retargeter:
    """Stateful per-session wrapper: owns the TargetFilter and the
    previous-yaw hold values, so callers just do
    `result = retargeter.step(landmarks_world_m, frame_id, dt, lowstate_q)`
    once per frame.
    """

    def __init__(self, min_visibility=0.5, alpha=0.3, yaw_clamp=math.radians(60)):
        self.min_visibility = min_visibility
        self.yaw_clamp = yaw_clamp
        self.filter = TargetFilter(alpha=alpha)
        self._prev_yaw = {"left": 0.0, "right": 0.0}

    def step(self, landmarks_world_m, frame_id, dt, lowstate_q=None):
        xyz_mp = landmarks_to_xyz(landmarks_world_m)
        xyz_robot = mediapipe_to_robot(xyz_mp)
        _origin, R_torso = torso_frame(xyz_robot)

        raw_targets = {}
        updated_sides = set()
        for side, idx in _SIDE_LANDMARKS.items():
            if not landmark_ok(landmarks_world_m, side, self.min_visibility):
                continue  # (R7) hold: don't touch this side's joints this tick

            shoulder_xyz = xyz_robot[idx["shoulder"]]
            elbow_xyz = xyz_robot[idx["elbow"]]
            wrist_xyz = xyz_robot[idx["wrist"]]

            upper_world = elbow_xyz - shoulder_xyz
            fore_world = wrist_xyz - elbow_xyz
            upper_torso = R_torso @ upper_world
            fore_torso = R_torso @ fore_world

            pitch, roll, yaw, elbow = arm_angles(
                upper_torso, fore_torso, side,
                prev_yaw=self._prev_yaw[side], yaw_clamp=self.yaw_clamp,
            )
            self._prev_yaw[side] = yaw

            names = _SIDE_ARM_JOINTS[side]
            raw_targets[names[0]] = pitch
            raw_targets[names[1]] = roll
            raw_targets[names[2]] = yaw
            raw_targets[names[3]] = elbow
            updated_sides.add(side)

        filtered = self.filter.step(raw_targets, dt)

        q = {}
        for side, names in _SIDE_ARM_JOINTS.items():
            for name in names:
                q[name] = filtered.get(name, self.filter.last_sent.get(name, 0.0))
        for side, names in _SIDE_WRIST_JOINTS.items():
            for name in names:
                q[name] = 0.0  # wrist roll/pitch/yaw: not derived from vision, always 0

        for name, lim in G1_29DOF_JOINT_LIMITS.items():
            if lim.index < 15:  # legs (0-11) and waist (12-14): hold, not retargeted
                if lowstate_q is not None:
                    q[name] = float(lowstate_q[lim.index])
                else:
                    q[name] = 0.0

        q_array = np.zeros(29, dtype=float)
        for name, lim in G1_29DOF_JOINT_LIMITS.items():
            q_array[lim.index] = q[name]

        return RetargetResult(q=q, q_array=q_array, updated_sides=updated_sides, frame_id=frame_id)
