'''
g1_joint_limits.py

Joint limits (position, velocity, torque) for the Unitree G1, 29-DOF rev 1.0,
copied from Unitree's own URDF (unitree_ros: robots/g1_description/g1_29dof_rev_1_0.urdf).
Index i here is the same motor as index i in unitree_sdk2_python's G1JointIndex.

Only valid for the 29-DOF G1 without hands. For a 23-DOF or with-hands unit,
use the matching URDF from the same repo and regenerate the table.

Units: position in rad, velocity in rad/s, effort (torque) in N*m.
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from dataclasses import dataclass


#----------------------------------------------------------
# Joint limit table
#----------------------------------------------------------
@dataclass(frozen=True)
class JointLimit:
    index: int
    lower: float      # rad
    upper: float      # rad
    velocity: float   # rad/s, max |joint velocity|
    effort: float      # N*m, max |joint torque|


# name -> JointLimit. Names match the *_joint names used in the URDF/MJCF
# and, minus the trailing "_joint", the human-readable G1JointIndex names.
G1_29DOF_JOINT_LIMITS = {
    "left_hip_pitch_joint":     JointLimit(0,  -2.5307,        2.8798,       32, 88),
    "left_hip_roll_joint":      JointLimit(1,  -0.5236,        2.9671,       20, 139),
    "left_hip_yaw_joint":       JointLimit(2,  -2.7576,        2.7576,       32, 88),
    "left_knee_joint":          JointLimit(3,  -0.087267,      2.8798,       20, 139),
    "left_ankle_pitch_joint":   JointLimit(4,  -0.87267,       0.5236,       30, 35),
    "left_ankle_roll_joint":    JointLimit(5,  -0.2618,        0.2618,       30, 35),
    "right_hip_pitch_joint":    JointLimit(6,  -2.5307,        2.8798,       32, 88),
    "right_hip_roll_joint":     JointLimit(7,  -2.9671,        0.5236,       20, 139),
    "right_hip_yaw_joint":      JointLimit(8,  -2.7576,        2.7576,       32, 88),
    "right_knee_joint":         JointLimit(9,  -0.087267,      2.8798,       20, 139),
    "right_ankle_pitch_joint":  JointLimit(10, -0.87267,       0.5236,       30, 35),
    "right_ankle_roll_joint":   JointLimit(11, -0.2618,        0.2618,       30, 35),
    "waist_yaw_joint":          JointLimit(12, -2.618,         2.618,        32, 88),
    "waist_roll_joint":         JointLimit(13, -0.52,          0.52,         30, 35),
    "waist_pitch_joint":        JointLimit(14, -0.52,          0.52,         30, 35),
    "left_shoulder_pitch_joint":JointLimit(15, -3.0892,        2.6704,       37, 25),
    "left_shoulder_roll_joint": JointLimit(16, -1.5882,        2.2515,       37, 25),
    "left_shoulder_yaw_joint":  JointLimit(17, -2.618,         2.618,        37, 25),
    "left_elbow_joint":         JointLimit(18, -1.0472,        2.0944,       37, 25),
    "left_wrist_roll_joint":    JointLimit(19, -1.972222054,   1.972222054,  37, 25),
    "left_wrist_pitch_joint":   JointLimit(20, -1.614429558,   1.614429558,  22, 5),
    "left_wrist_yaw_joint":     JointLimit(21, -1.614429558,   1.614429558,  22, 5),
    "right_shoulder_pitch_joint":JointLimit(22,-3.0892,        2.6704,       37, 25),
    "right_shoulder_roll_joint":JointLimit(23, -2.2515,        1.5882,       37, 25),
    "right_shoulder_yaw_joint": JointLimit(24, -2.618,         2.618,        37, 25),
    "right_elbow_joint":        JointLimit(25, -1.0472,        2.0944,       37, 25),
    "right_wrist_roll_joint":   JointLimit(26, -1.972222054,   1.972222054,  37, 25),
    "right_wrist_pitch_joint":  JointLimit(27, -1.614429558,   1.614429558,  22, 5),
    "right_wrist_yaw_joint":    JointLimit(28, -1.614429558,   1.614429558,  22, 5),
}

EXPECTED_DOF_COUNT = 29  # legs 6*2 + waist 3 + arms 7*2. No hands in this variant.


#----------------------------------------------------------
# Checking and clamping helpers
#----------------------------------------------------------
def check_dof_count(joint_names) -> list:
    '''Compare a list/iterable of joint names against the expected 29 DOF.

    Returns a list of human-readable problem strings; empty list = OK.
    Catches a silent mismatch, e.g. someone swaps in the 23-DOF or
    with-hands model and the rest of the code still assumes 29 plain joints.
    '''
    problems = []
    given = list(joint_names)

    # wrong number of joints
    if len(given) != EXPECTED_DOF_COUNT:
        problems.append(
            f"expected {EXPECTED_DOF_COUNT} DOF, got {len(given)} "
            f"(check whether this is really the 29-DOF-no-hands G1 variant)"
        )

    # joints that are missing from, or not part of, the expected table
    given_set = set(given)
    expected_set = set(G1_29DOF_JOINT_LIMITS.keys())
    missing = expected_set - given_set
    extra = given_set - expected_set
    if missing:
        problems.append(f"missing joints: {sorted(missing)}")
    if extra:
        problems.append(f"unrecognised joints (not in the 29-DOF table): {sorted(extra)}")
    return problems


def clamp_position(joint_name: str, value: float) -> float:
    '''Clamp an angle (rad) into the joint's position range.'''
    lim = G1_29DOF_JOINT_LIMITS[joint_name]
    return min(max(value, lim.lower), lim.upper)


def clamp_velocity(joint_name: str, value: float) -> float:
    '''Clamp a speed (rad/s) into the joint's +/- velocity limit.'''
    lim = G1_29DOF_JOINT_LIMITS[joint_name]
    return min(max(value, -lim.velocity), lim.velocity)


def validate_command(positions: dict, velocities: dict = None, tol: float = 1e-6) -> list:
    '''Check a {joint_name: value} position dict (and optional velocity dict)
    against the real hardware limits. Returns a list of violation strings;
    empty list = every value is within spec.

    A last-line check before sending a command to the robot. Catches cases
    IK-time limiting can miss (smoothing, interpolation, network jitter).
    '''
    violations = []

    # positions
    for name, value in positions.items():
        lim = G1_29DOF_JOINT_LIMITS.get(name)
        if lim is None:
            violations.append(f"{name}: not a recognised G1 29-DOF joint")
            continue
        if value < lim.lower - tol or value > lim.upper + tol:
            violations.append(
                f"{name}: position {value:.4f} rad outside [{lim.lower:.4f}, {lim.upper:.4f}] rad"
            )

    # velocities (optional)
    if velocities:
        for name, value in velocities.items():
            lim = G1_29DOF_JOINT_LIMITS.get(name)
            if lim is None:
                continue
            if abs(value) > lim.velocity + tol:
                violations.append(
                    f"{name}: |velocity| {abs(value):.4f} rad/s exceeds {lim.velocity} rad/s"
                )
    return violations


# ---------------------------------------------------------------------------
# Notes on GMR's own joint ranges (assets/unitree_g1/g1_mocap_29dof.xml)
# ---------------------------------------------------------------------------
# GMR's retargeting MJCF uses the SAME 29 joints in the SAME order, but with
# its own <joint range="..."> values, which are a safe subset of the table
# above on every joint checked (e.g. left_hip_pitch is clamped to
# [-1.57, 1.57] in GMR's MJCF vs the real [-2.5307, 2.8798] on hardware) -
# this is deliberate and fine: mink.ConfigurationLimit enforces the MJCF's
# range as a hard IK constraint every solve, so IK output can never exceed
# GMR's ranges, and GMR's ranges never exceed hardware. Left as-is.
#
# What GMR's MJCF does NOT define correctly is per-joint VELOCITY:
#   - GeneralMotionRetargeting(..., use_velocity_limit=False) by default
#     (general_motion_retargeting/motion_retarget.py) - if the retargeting
#     call site doesn't pass use_velocity_limit=True explicitly, IK solves
#     with NO velocity constraint at all.
#   - When enabled, GMR applies one flat 3*pi rad/s (~9.42 rad/s) bound to
#     every one of the 29 motors, not the per-joint values above (20-37
#     rad/s depending on joint). That's conservative (never over the real
#     max anywhere) but not an accurate model of the robot - a fast arm
#     swing is capped well below what the arm can actually do, and it is
#     not what "match the model limitations" asks for.
#
# To get IK-time velocity limiting that actually reflects the hardware,
# build mink's VelocityLimit dict from this file instead of the flat bound:
#
#   from g1_joint_limits import G1_29DOF_JOINT_LIMITS
#   VELOCITY_LIMITS = {name: lim.velocity for name, lim in G1_29DOF_JOINT_LIMITS.items()}
#   self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))
#
# This still only covers the IK solve itself. Whatever assembles the final
# LowCmd sent over unitree_sdk2py (interpolation, frame-rate conversion,
# any smoothing) should also run validate_command() on the actual per-tick
# command before it goes out - that is the only check downstream of IK.