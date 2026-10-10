'''
pose_to_target.py

Converts MediaPipe body landmarks into the targets GMR needs to move the G1's arms.

Two stages, kept separate so visibility gating (pose_filtering.py) can sit between them:
    1. measure_segments(): 33 landmarks -> body-relative segment data
       (torso orientation, upper-arm and forearm directions, which are visible)
    2. TargetBuilder.build(): segments -> {body name: (position, quaternion)}
       in the form GMR's retarget() expects

Arm directions are copied from the person and combined with the robot's own
bone lengths, so the person's height does not affect the joint angles.

pip mujoco is used here only as a maths library (forward kinematics on the G1 model).

This file used to be called: mediapipe_to_gmr.py
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from dataclasses import dataclass, field

import numpy as np


#----------------------------------------------------------
# Landmark indices and constants
#----------------------------------------------------------
# MediaPipe pose landmark indices used here.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

# Left/right landmark pairs, used to swap sides in mirror mode.
MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [(i, i + 1) for i in range(11, 33, 2)]

SIDES = ("left", "right")
ARM_LANDMARKS = {
    "left": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right": (R_SHOULDER, R_ELBOW, R_WRIST),
}
# Landmarks each gated segment depends on.
SEGMENT_LANDMARKS = {
    "torso": (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP),
    "left_upper": (L_SHOULDER, L_ELBOW),
    "left_fore": (L_ELBOW, L_WRIST),
    "right_upper": (R_SHOULDER, R_ELBOW),
    "right_fore": (R_ELBOW, R_WRIST),
}

UP = np.array([0.0, 0.0, 1.0])
DOWN = -UP

# Elbow bend (deg) below which the elbow hinge axis, and so upper-arm twist,
# cannot be measured. Between the two values the measured hinge is blended
# with a reference hinge.
HINGE_BLEND_DEG = (8.0, 30.0)

# Converts MediaPipe world axes (x right, y down, z away from camera) to robot
# world axes (x forward, y left, z up): robot = (-z_mp, x_mp, -y_mp).
_MP_TO_ROBOT = np.array([
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


#----------------------------------------------------------
# Landmark helpers
#----------------------------------------------------------
def landmarks_to_array(landmarks):
    '''Accept a list of {x,y,z,visibility} dicts, MediaPipe landmark objects,
    or an (33, 3|4) array; return a float (33, 4) array [x, y, z, visibility].'''
    # already a NumPy array: check the shape, add visibility = 1 if it is missing
    if isinstance(landmarks, np.ndarray):
        arr = np.asarray(landmarks, dtype=float)
        if arr.shape == (33, 3):
            arr = np.hstack([arr, np.ones((33, 1))])
        if arr.shape != (33, 4):
            raise ValueError(f"expected (33, 3) or (33, 4) landmarks, got {arr.shape}")
        return arr.copy()

    if landmarks is None or len(landmarks) != 33:
        raise ValueError("expected exactly 33 landmarks")

    # list of dicts or MediaPipe landmark objects: pull out x, y, z, visibility
    rows = []
    for lm in landmarks:
        if isinstance(lm, dict):
            rows.append((lm["x"], lm["y"], lm["z"], lm.get("visibility", 1.0)))
        else:
            rows.append((lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)))
    arr = np.array(rows, dtype=float)

    # a missing or nan visibility counts as 0 (not visible)
    if arr[:, 3].dtype != float or np.any(~np.isfinite(arr[:, 3])):
        arr[:, 3] = np.nan_to_num(arr[:, 3], nan=0.0)
    return arr


def mediapipe_to_robot(points):
    '''MediaPipe world (x right, y down, z away) -> robot world (x fwd, y left, z up).
    Works on a single (3,) point or an (N, 3) batch.'''
    return np.asarray(points, dtype=float) @ _MP_TO_ROBOT.T


def mirror_landmarks(arr):
    '''Swap left/right landmarks and reflect across the body's sagittal plane
    (MediaPipe x), so the robot moves like the person's mirror image.'''
    out = arr.copy()
    for a, b in MIRROR_PAIRS:
        out[[a, b]] = out[[b, a]]   # swap the two sides
    out[:, 0] = -out[:, 0]          # flip left-right
    return out


#----------------------------------------------------------
# Maths helpers (vectors and rotations)
#----------------------------------------------------------
def _unit(v, eps=1e-9):
    '''Normalise v to length 1, or None if it is too short / not finite.'''
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return None
    return v / n


def _frame_from_lateral_up(lateral, up):
    '''Right-handed frame [x fwd, y lateral(left), z up] from a left-pointing
    lateral vector and an approximate up vector. None if degenerate.'''
    up = _unit(up)
    lateral = _unit(lateral)
    if up is None or lateral is None:
        return None
    x = _unit(np.cross(lateral, up))    # forward is perpendicular to both
    if x is None:
        return None
    y = np.cross(up, x)                 # re-derive lateral so all three are perpendicular
    return np.column_stack([x, y, up])


def rot_about(axis, angle):
    '''Rotation matrix for a rotation of `angle` rad about `axis` (Rodrigues' formula).'''
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def min_rotation(a, b, fallback_axis):
    '''Smallest rotation taking unit vector a onto unit vector b. For
    antiparallel vectors rotate 180 deg about fallback_axis.'''
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))     # cos of the angle between a and b
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)                        # sin of the angle
    if s < 1e-6:
        # a and b are parallel (nothing to do) or opposite (pick any axis)
        if c > 0:
            return np.eye(3)
        return rot_about(fallback_axis, np.pi)
    return rot_about(axis / s, np.arctan2(s, c))


def waist_euler_from_matrix(r):
    '''G1 waist chain is yaw(z) -> roll(x) -> pitch(y): R = Rz(yaw) Rx(roll) Ry(pitch).
    Returns (yaw, roll, pitch).'''
    roll = np.arcsin(np.clip(r[2, 1], -1.0, 1.0))
    yaw = np.arctan2(-r[0, 1], r[1, 1])
    pitch = np.arctan2(-r[2, 0], r[2, 2])
    return yaw, roll, pitch


def waist_matrix(yaw, roll, pitch):
    '''Inverse of waist_euler_from_matrix: three waist angles -> rotation matrix.'''
    return (rot_about(np.array([0.0, 0, 1]), yaw)
            @ rot_about(np.array([1.0, 0, 0]), roll)
            @ rot_about(np.array([0.0, 1, 0]), pitch))


def matrix_to_quat_wxyz(r):
    '''Rotation matrix -> unit quaternion (w, x, y, z).'''
    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    # four cases, picking the numerically safest one for this matrix
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


#----------------------------------------------------------
# Segments (output of stage 1)
#----------------------------------------------------------
@dataclass
class Segments:
    '''Body-relative pose of the leader.

    torso: 3x3 rotation of the torso frame relative to the pelvis frame.
    upper/fore: {side: unit vector} in the person's torso frame (x fwd, y left, z up).
    valid: {segment name: bool} measured visibility verdict (see SEGMENT_LANDMARKS).
    '''
    torso: np.ndarray = field(default_factory=lambda: np.eye(3))
    upper: dict = field(default_factory=lambda: {s: DOWN.copy() for s in SIDES})
    fore: dict = field(default_factory=lambda: {s: DOWN.copy() for s in SIDES})
    valid: dict = field(default_factory=lambda: {k: False for k in SEGMENT_LANDMARKS})

    def copy(self):
        '''Independent copy (so filters can change it without touching the original).'''
        return Segments(
            torso=self.torso.copy(),
            upper={s: v.copy() for s, v in self.upper.items()},
            fore={s: v.copy() for s, v in self.fore.items()},
            valid=dict(self.valid),
        )


def neutral_segments():
    '''Upright torso, both arms hanging straight down.'''
    return Segments()


def segment_validity(arr, min_visibility):
    '''{segment name: True} if every landmark that segment needs is visible enough.'''
    vis = arr[:, 3]
    return {name: bool(np.all(vis[list(idx)] >= min_visibility)) for name, idx in SEGMENT_LANDMARKS.items()}


#----------------------------------------------------------
# Stage 1: landmarks -> segments
#----------------------------------------------------------
def measure_segments(landmarks, min_visibility=0.5, mirror=False):
    '''33 MediaPipe world landmarks -> Segments (plus per-segment validity).

    Frames:
    * pelvis: lateral axis from the hip line, projected horizontal; z = world up.
      If the hips are not visible the shoulder line is used instead (upper-body
      framing is the common webcam case, and MediaPipe hallucinates hips there).
    * torso: lateral axis from the shoulder line, up axis from hip centre to
      shoulder centre (world up if hips are not visible, so torso lean is
      then unobserved and the torso segment is reported invalid).
    Arm directions are expressed in the person's torso frame.
    '''
    arr = landmarks_to_array(landmarks)
    if mirror:
        arr = mirror_landmarks(arr)
    if not np.all(np.isfinite(arr[:, :3])):
        raise ValueError("non-finite landmark coordinates")

    valid = segment_validity(arr, min_visibility)
    p = mediapipe_to_robot(arr[:, :3])      # all positions in robot axes

    # lines across the shoulders and hips, and the spine direction
    shoulder_line = p[L_SHOULDER] - p[R_SHOULDER]
    hips_ok = bool(np.all(arr[[L_HIP, R_HIP], 3] >= min_visibility))
    if hips_ok:
        hip_line = p[L_HIP] - p[R_HIP]
        spine = 0.5 * (p[L_SHOULDER] + p[R_SHOULDER]) - 0.5 * (p[L_HIP] + p[R_HIP])
    else:
        # hips not visible: fall back to the shoulder line and world up
        hip_line = shoulder_line
        spine = UP

    # pelvis frame (kept horizontal) and torso frame (follows the spine)
    horiz = hip_line - np.dot(hip_line, UP) * UP
    r_pelvis = _frame_from_lateral_up(horiz, UP)
    r_torso = _frame_from_lateral_up(shoulder_line, spine)
    if r_pelvis is None or r_torso is None:
        raise ValueError("degenerate hip/shoulder geometry")

    # torso orientation relative to the pelvis
    seg = Segments(torso=r_pelvis.T @ r_torso, valid=valid)
    if not hips_ok:
        seg.valid["torso"] = False

    # arm directions, expressed in the torso frame
    for side in SIDES:
        s_idx, e_idx, w_idx = ARM_LANDMARKS[side]
        u = _unit(p[e_idx] - p[s_idx])      # shoulder -> elbow
        f = _unit(p[w_idx] - p[e_idx])      # elbow -> wrist
        if u is None:
            seg.valid[f"{side}_upper"] = False
        else:
            seg.upper[side] = r_torso.T @ u
        if f is None:
            seg.valid[f"{side}_fore"] = False
        else:
            seg.fore[side] = r_torso.T @ f
    return seg


#----------------------------------------------------------
# Robot rest geometry
#----------------------------------------------------------
@dataclass
class RobotRest:
    '''G1 geometry at q = 0, read from the MJCF that GMR solves on.'''
    pelvis_pos: np.ndarray
    u0: dict            # rest upper-arm unit vector per side (world = link frame at q=0)
    f0: dict            # rest forearm unit vector per side
    v0: dict            # shoulder_roll -> elbow vector per side (m)
    w0: dict            # elbow -> wrist_roll vector per side (m)
    n0: dict            # rest elbow hinge (unit, f0 x u0 normalised)

    @classmethod
    def from_model(cls, model):
        # imported here (not at the top): MuJoCo is slow to load, only needed when this runs
        import mujoco

        # put the robot in its zero pose and read where every body is
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, data)
        pos = {model.body(i).name: data.xpos[i].copy() for i in range(model.nbody)}

        u0, f0, v0, w0, n0 = {}, {}, {}, {}, {}
        for side in SIDES:
            s = pos[f"{side}_shoulder_roll_link"]
            e = pos[f"{side}_elbow_link"]
            w = pos[f"{side}_wrist_roll_link"]
            v0[side], w0[side] = e - s, w - e                   # bone vectors (with length)
            u0[side], f0[side] = _unit(e - s), _unit(w - e)     # bone directions
            n0[side] = _unit(np.cross(f0[side], u0[side]))      # rest elbow hinge axis
        return cls(pos["pelvis"].copy(), u0, f0, v0, w0, n0)


#----------------------------------------------------------
# Elbow hinge
#----------------------------------------------------------
def _link_frame(axis, hinge):
    '''Orthonormal basis [axis, hinge, axis x hinge] (hinge re-orthogonalised).'''
    hinge = hinge - np.dot(hinge, axis) * axis
    hinge = hinge / np.linalg.norm(hinge)
    return np.column_stack([axis, hinge, np.cross(axis, hinge)])


_FWD = np.array([1.0, 0.0, 0.0])
_LAT = np.array([0.0, 1.0, 0.0])


def reference_hinge(u_torso):
    '''Elbow hinge to assume when the arm is straight (twist unobservable),
    for upper-arm direction u in the torso frame.

    The lateral axis carried along by the smallest rotation from "arm straight
    forward" to u. This matches the G1's zero-yaw hinge for arms down, forward
    and overhead, and is continuous everywhere except arm straight back,
    which the G1 cannot reach (shoulder pitch stops at 66 deg back).'''
    return min_rotation(_FWD, u_torso, _LAT) @ _LAT


def hinge_axis(u, f, n_ref):
    '''Elbow hinge axis for upper-arm direction u and forearm direction f.

    Returns (hinge, confidence). Confidence is 0 for a straight arm (hinge
    unobservable, n_ref used) rising to 1 once the elbow is bent by
    HINGE_BLEND_DEG[1] degrees.'''
    # the measured hinge is perpendicular to both bones; the bend angle tells how reliable it is
    cross = np.cross(f, u)
    s = np.linalg.norm(cross)
    bend = np.degrees(np.arctan2(s, np.dot(f, u)))
    lo, hi = HINGE_BLEND_DEG
    conf = float(np.clip((bend - lo) / (hi - lo), 0.0, 1.0))

    # reference hinge, made perpendicular to the upper arm
    n_ref = n_ref - np.dot(n_ref, u) * u
    n_ref = n_ref / np.linalg.norm(n_ref)
    if conf <= 0.0:
        return n_ref, 0.0       # straight arm: only the reference is usable

    # blend measured and reference hinges by confidence
    n_meas = cross / s
    n = conf * n_meas + (1.0 - conf) * n_ref
    n = n - np.dot(n, u) * u
    norm = np.linalg.norm(n)
    if norm < 1e-3:
        return n_meas, conf     # the two cancelled out: use the measured one
    return n / norm, conf


#----------------------------------------------------------
# Hip clearance
#----------------------------------------------------------
# Minimum outward angle (deg) of an arm hanging beside the body. Measured on
# g1_29dof.xml: with both upper arm and forearm vertical the hand geometry
# overlaps the hip links by 3.9 cm; ~8 deg of outward angle clears them.
HIP_CLEARANCE_DEG = 9.0


def _with_min_outward(v, side, min_deg):
    '''Rotate direction v (torso frame) about the torso x axis so that its
    outward angle from vertical-down is at least min_deg, applied fully when
    v points down beside the body and faded out as it points forward/back
    (the hand then clears the hip anyway).'''
    sign = 1.0 if side == "left" else -1.0      # outward is +y on the left, -y on the right
    down = -v[2]
    if down <= 0.5:
        return v                                # not hanging down: leave it
    # how strongly to apply the minimum angle (1 = fully, 0 = not at all)
    fade = float(np.clip(1.0 - abs(v[0]) / 0.4, 0.0, 1.0)) * float(np.clip((down - 0.5) / 0.3, 0.0, 1.0))
    need = np.radians(min_deg) * fade
    lateral = sign * v[1]
    planar = np.hypot(lateral, -v[2])
    current = np.arctan2(lateral, -v[2])        # current outward angle
    if current >= need or planar < 1e-9:
        return v                                # already far enough out
    return np.array([v[0], sign * planar * np.sin(need), -planar * np.cos(need)])


def hip_clearance(side, u, f, min_deg=HIP_CLEARANCE_DEG):
    '''Keep a hanging arm's hand off the G1's hip links. The forearm rule only
    applies when the upper arm also hangs (hand beside the hip); a forearm
    reaching inward from a raised elbow (hand on hip, arms crossed) is left
    alone.'''
    u2 = _with_min_outward(u, side, min_deg)
    f2 = _with_min_outward(f, side, min_deg) if -u[2] > 0.8 else f
    return u2, f2


#----------------------------------------------------------
# Stage 2: segments -> GMR targets
#----------------------------------------------------------
class TargetBuilder:
    '''Segments -> GMR human_data for data/mediapipe_to_g1.json.

    waist: "3dof" (yaw, roll, pitch), "yaw" (roll/pitch held at 0, for
    waist-locked robots) or "off" (torso held upright).
    '''

    def __init__(self, model, waist="off", waist_limits=None):
        # imported here (not at the top): MuJoCo is slow to load, only needed when this runs
        import mujoco

        self.model = model
        self.data = mujoco.MjData(model)
        self._mujoco = mujoco
        self.rest = RobotRest.from_model(model)
        if waist not in ("3dof", "yaw", "off"):
            raise ValueError(f"unknown waist mode {waist!r}")
        self.waist = waist
        # position of each waist joint in the model's qpos array
        self.waist_joint_adr = {
            j: model.joint(f"waist_{j}_joint").qposadr[0] for j in ("yaw", "roll", "pitch")
        }
        # waist limits default to the model's own joint ranges
        if waist_limits is None:
            waist_limits = {j: tuple(model.joint(f"waist_{j}_joint").range) for j in ("yaw", "roll", "pitch")}
        self.waist_limits = waist_limits
        self.shoulder_body = {s: model.body(f"{s}_shoulder_roll_link").id for s in SIDES}
        self.torso_body = model.body("torso_link").id
        self.reset()

    def reset(self):
        '''Forget the elbow-hinge history (start of a new session).'''
        self.prev_hinge = {s: None for s in SIDES}

    def _fallback_hinge(self, side, u_t):
        '''Hinge to use while twist is unobservable (straight arm), torso frame.

        The previous frame's hinge, re-orthogonalised to the new upper-arm
        direction (parallel transport), so the arm keeps whatever twist it had
        when it straightened instead of snapping to a different one. The
        geometric reference_hinge is only used with no usable history.'''
        prev = self.prev_hinge[side]
        if prev is not None:
            n = prev - np.dot(prev, u_t) * u_t
            norm = np.linalg.norm(n)
            if norm > 0.3:
                return n / norm
        return reference_hinge(u_t)

    def clamp_torso(self, r_rel):
        '''Torso rotation -> {yaw, roll, pitch}, zeroed per the waist mode and limited to the joint ranges.'''
        yaw, roll, pitch = waist_euler_from_matrix(r_rel)
        if self.waist == "off":
            yaw = roll = pitch = 0.0
        elif self.waist == "yaw":
            roll = pitch = 0.0
        angles = {"yaw": yaw, "roll": roll, "pitch": pitch}
        clamped = {j: float(np.clip(a, *self.waist_limits[j])) for j, a in angles.items()}
        return clamped

    def build(self, seg):
        '''Returns (human_data, rot_weight_scale, info).

        rot_weight_scale: {human body name: factor in [0, 1]} to multiply the
        config's rotation weight by for this frame (twist confidence).
        '''
        # torso orientation, limited to what the waist can do
        waist = self.clamp_torso(seg.torso)
        r_torso = waist_matrix(waist["yaw"], waist["roll"], waist["pitch"])

        # Forward kinematics of the waist alone gives the robot's shoulder and
        # torso poses for this torso orientation (arm joints at zero).
        d = self.data
        d.qpos[:] = self.model.qpos0
        for j, adr in self.waist_joint_adr.items():
            d.qpos[adr] = waist[j]
        self._mujoco.mj_kinematics(self.model, d)

        # pelvis and torso targets
        ident = np.array([1.0, 0.0, 0.0, 0.0])     # "no rotation" quaternion
        human = {
            "pelvis": (self.rest.pelvis_pos.copy(), ident.copy()),
            "torso": (d.xpos[self.torso_body].copy(), matrix_to_quat_wxyz(r_torso)),
        }
        weight_scale = {}
        info = {"waist": waist, "hinge_conf": {}}

        # arm targets, one side at a time
        for side in SIDES:
            # person's arm directions, adjusted so the hand clears the hip, then moved into the world frame
            u_t, f_t = hip_clearance(side, seg.upper[side], seg.fore[side])
            u = r_torso @ u_t
            f = r_torso @ f_t
            u0, f0, n0 = self.rest.u0[side], self.rest.f0[side], self.rest.n0[side]

            # elbow hinge (remembered for the next frame in case the arm straightens)
            n, conf = hinge_axis(u, f, r_torso @ self._fallback_hinge(side, u_t))
            self.prev_hinge[side] = r_torso.T @ n

            # rotation of each arm link from its rest pose
            r_upper = _link_frame(u, n) @ _link_frame(u0, n0).T
            r_fore = _link_frame(f, n) @ _link_frame(f0, n0).T

            # rebuild the arm from the robot's own bone lengths
            shoulder = d.xpos[self.shoulder_body[side]].copy()
            elbow = shoulder + r_upper @ self.rest.v0[side]
            wrist = elbow + r_fore @ self.rest.w0[side]

            human[f"{side}_upper_arm"] = (shoulder + 0.5 * (elbow - shoulder), matrix_to_quat_wxyz(r_upper))
            human[f"{side}_elbow"] = (elbow, matrix_to_quat_wxyz(r_fore))
            human[f"{side}_wrist"] = (wrist, ident.copy())
            # Twist is only observable with a bent elbow; keep some weight so
            # the arm settles on the reference twist instead of drifting.
            weight_scale[f"{side}_upper_arm"] = 0.3 + 0.7 * conf
            weight_scale[f"{side}_elbow"] = 0.3 + 0.7 * conf
            info["hinge_conf"][side] = conf
        return human, weight_scale, info