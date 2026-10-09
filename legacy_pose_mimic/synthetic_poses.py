"""
Synthetic 33-landmark MediaPipe world poses with known ground truth.

Each pose is specified as limb directions in the person's own torso frame
(x forward, y person-left, z up) plus a torso orientation relative to the
pelvis, then rendered into MediaPipe world coordinates (metres, origin at the
hip centre, x image-right, y down, z away from camera) for a person facing
the camera, optionally turned by `yaw_deg` about the vertical.

The ground truth (the directions the robot should reproduce) is kept on the
returned Pose so tests never have to re-derive it from the landmarks.
"""

from dataclasses import dataclass

import numpy as np

HIP_HALF = 0.10
SHOULDER_HALF = 0.18
SPINE = 0.50
UPPER_ARM = 0.28
FOREARM = 0.26
THIGH = 0.42
SHIN = 0.40


def unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def robot_to_mediapipe(p):
    """Inverse of mediapipe_to_gmr.mediapipe_to_robot: (x fwd, y left, z up)
    -> (x image-right, y down, z away from camera)."""
    p = np.asarray(p, dtype=float)
    return np.stack([p[..., 1], -p[..., 2], -p[..., 0]], axis=-1)


def mirror_dir(v):
    """Same direction for the right side (reflect y)."""
    return np.array([v[0], -v[1], v[2]])


@dataclass
class Pose:
    name: str
    landmarks: np.ndarray        # (33, 4) MediaPipe world + visibility
    upper: dict                  # side -> unit dir in person torso frame
    fore: dict
    torso: np.ndarray            # 3x3 torso-relative-to-pelvis rotation

    def as_dicts(self):
        return [{"x": float(r[0]), "y": float(r[1]), "z": float(r[2]), "visibility": float(r[3])}
                for r in self.landmarks]


def build_pose(name, left_upper, left_fore, right_upper=None, right_fore=None,
               torso=None, yaw_deg=0.0, visibility=0.99):
    """Directions are in the person's torso frame. Right-side directions
    default to the mirror of the left side's."""
    torso = np.eye(3) if torso is None else np.asarray(torso, dtype=float)
    lu, lf = unit(*left_upper), unit(*left_fore)
    ru = unit(*right_upper) if right_upper is not None else mirror_dir(lu)
    rf = unit(*right_fore) if right_fore is not None else mirror_dir(lf)

    p = np.zeros((33, 3))
    l_hip, r_hip = np.array([0, HIP_HALF, 0.0]), np.array([0, -HIP_HALF, 0.0])
    sh_mid = torso @ np.array([0, 0, SPINE])
    l_sh = sh_mid + torso @ np.array([0, SHOULDER_HALF, 0])
    r_sh = sh_mid + torso @ np.array([0, -SHOULDER_HALF, 0])
    l_el = l_sh + torso @ lu * UPPER_ARM
    r_el = r_sh + torso @ ru * UPPER_ARM
    l_wr = l_el + torso @ lf * FOREARM
    r_wr = r_el + torso @ rf * FOREARM

    p[11], p[12], p[13], p[14], p[15], p[16] = l_sh, r_sh, l_el, r_el, l_wr, r_wr
    p[23], p[24] = l_hip, r_hip
    # Legs straight down, feet forward.
    for hip_i, knee_i, ank_i, heel_i, toe_i, hip in ((23, 25, 27, 29, 31, l_hip), (24, 26, 28, 30, 32, r_hip)):
        p[knee_i] = hip + np.array([0, 0, -THIGH])
        p[ank_i] = p[knee_i] + np.array([0, 0, -SHIN])
        p[heel_i] = p[ank_i] + np.array([-0.05, 0, -0.05])
        p[toe_i] = p[ank_i] + np.array([0.15, 0, -0.07])
    # Head: nose in front of the neck, eyes/ears around it.
    head = sh_mid + torso @ np.array([0, 0, 0.22])
    p[0] = head + torso @ np.array([0.10, 0, 0])
    for i, off in zip(range(1, 11), [(0.08, 0.03, 0.03), (0.08, 0.035, 0.03), (0.08, 0.045, 0.03),
                                      (0.08, -0.03, 0.03), (0.08, -0.035, 0.03), (0.08, -0.045, 0.03),
                                      (0.0, 0.07, 0.02), (0.0, -0.07, 0.02),
                                      (0.09, 0.02, -0.04), (0.09, -0.02, -0.04)]):
        p[i] = head + torso @ np.array(off)
    # Hands: pinky/index/thumb a few cm past the wrist along the forearm.
    for wr_i, fore, (pinky, index, thumb) in ((15, torso @ lf, (17, 19, 21)), (16, torso @ rf, (18, 20, 22))):
        p[pinky] = p[wr_i] + fore * 0.08
        p[index] = p[wr_i] + fore * 0.09
        p[thumb] = p[wr_i] + fore * 0.05

    p = p @ rot_z(yaw_deg).T
    mp = robot_to_mediapipe(p)
    arr = np.hstack([mp, np.full((33, 1), visibility)])
    return Pose(name, arr, {"left": lu, "right": ru}, {"left": lf, "right": rf}, torso)


DOWN = (0, 0, -1)
UP = (0, 0, 1)
FWD = (1, 0, 0)
LEFT = (0, 1, 0)


def base_poses():
    """The Step-3 pose set, person facing the camera."""
    return [
        build_pose("arms_down", DOWN, DOWN),
        build_pose("t_pose", LEFT, LEFT),
        build_pose("arms_forward", FWD, FWD),
        build_pose("arms_overhead", UP, UP),
        build_pose("left_up_right_down", UP, UP, DOWN, DOWN),
        build_pose("elbows_90_forearms_forward", DOWN, FWD),
        build_pose("goalpost", LEFT, UP),
        build_pose("forearms_crossed", (0.35, 0.15, -1), (0.35, -1, 0.45)),
        build_pose("hand_on_hip", (-0.35, 0.75, -0.6), (0.28, -0.62, -0.73), DOWN, DOWN),
        build_pose("wave", (0, 1, 0.3), (0, -0.25, 1), DOWN, DOWN),
        # Leaning: arms hang with gravity (vertical in the world), as a
        # person's relaxed arms do, i.e. rotated back by the lean in the
        # torso frame.
        build_pose("lean_forward", rot_y(25).T @ unit(*DOWN), rot_y(25).T @ unit(*DOWN), torso=rot_y(25)),
        build_pose("lean_sideways", rot_x(-18).T @ unit(0, 0.15, -1), rot_x(-18).T @ unit(*DOWN),
                   rot_x(-18).T @ unit(0, -0.15, -1), rot_x(-18).T @ unit(*DOWN), torso=rot_x(-18)),
        build_pose("torso_twisted", (0.3, 0.2, -1), FWD, torso=rot_z(30)),
    ]


def all_poses():
    poses = base_poses()
    rotated = []
    for pose in poses:
        rotated.append(_rebuild(pose, yaw_deg=30.0))
    return poses + rotated


def _rebuild(pose, yaw_deg):
    b = build_pose(pose.name + "_rot30", pose.upper["left"], pose.fore["left"],
                   pose.upper["right"], pose.fore["right"], torso=pose.torso, yaw_deg=yaw_deg)
    return b


def slerp_dir(a, b, s, via=(1.0, 0.0, 0.0)):
    """Rotate unit vector a toward b by fraction s along the great circle.
    Nearly opposite directions (arm up -> arm down) swing through `via`
    (forward), the way a person actually moves an arm, instead of passing
    through the shoulder as a straight landmark blend would."""
    a, b = unit(*a), unit(*b)
    if np.dot(a, b) < -0.9:
        mid = unit(*via)
        return slerp_dir(a, mid, 2 * s) if s < 0.5 else slerp_dir(mid, b, 2 * s - 1)
    ang = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    if ang < 1e-6:
        return a
    return (np.sin((1 - s) * ang) * a + np.sin(s * ang) * b) / np.sin(ang)


def blend(a, b, s, yaw_deg=0.0):
    """Pose part-way (s in [0, 1]) from pose a to pose b: limb directions and
    torso orientation are interpolated, so bones keep their length and move
    along arcs like a real person's."""
    from scipy.spatial.transform import Rotation, Slerp

    torso = Slerp([0, 1], Rotation.from_matrix(np.stack([a.torso, b.torso])))([s]).as_matrix()[0]
    return build_pose(f"{a.name}->{b.name}",
                      slerp_dir(a.upper["left"], b.upper["left"], s), slerp_dir(a.fore["left"], b.fore["left"], s),
                      slerp_dir(a.upper["right"], b.upper["right"], s), slerp_dir(a.fore["right"], b.fore["right"], s),
                      torso=torso, yaw_deg=yaw_deg)
