import math

import pytest

from retarget_upper_body import Retargeter
from pose_fixtures import (
    arm_straight_forward,
    arms_hanging,
    t_pose,
    t_pose_elbows_bent_90,
)

FIVE_DEG = math.radians(5)


def _q_deg(result, joint):
    return math.degrees(result.q[joint])


def test_arms_hanging_all_eight_joints_near_zero():
    r = Retargeter()
    result = r.step(arms_hanging(), frame_id=0, dt=1.0)
    for side in ("left", "right"):
        for suffix in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
            angle = result.q[f"{side}_{suffix}_joint"]
            assert abs(angle) <= FIVE_DEG, f"{side}_{suffix}_joint = {math.degrees(angle):.1f} deg"


def test_t_pose_shoulder_roll_and_elbow():
    r = Retargeter()
    result = r.step(t_pose(), frame_id=0, dt=1.0)
    assert _q_deg(result, "left_shoulder_roll_joint") == pytest.approx(90, abs=5)
    assert _q_deg(result, "right_shoulder_roll_joint") == pytest.approx(-90, abs=5)
    assert abs(_q_deg(result, "left_elbow_joint")) <= 5
    assert abs(_q_deg(result, "right_elbow_joint")) <= 5
    assert abs(_q_deg(result, "left_shoulder_pitch_joint")) <= 5
    assert abs(_q_deg(result, "right_shoulder_pitch_joint")) <= 5


def test_arm_straight_forward_shoulder_pitch():
    r = Retargeter()
    result = r.step(arm_straight_forward(), frame_id=0, dt=1.0)
    assert _q_deg(result, "left_shoulder_pitch_joint") == pytest.approx(-90, abs=5)
    assert _q_deg(result, "right_shoulder_pitch_joint") == pytest.approx(-90, abs=5)
    assert abs(_q_deg(result, "left_shoulder_roll_joint")) <= 5
    assert abs(_q_deg(result, "right_shoulder_roll_joint")) <= 5


def test_t_pose_elbow_bent_90():
    r = Retargeter()
    result = r.step(t_pose_elbows_bent_90(), frame_id=0, dt=1.0)
    assert _q_deg(result, "left_elbow_joint") == pytest.approx(90, abs=5)
    assert _q_deg(result, "right_elbow_joint") == pytest.approx(90, abs=5)
    # Shoulders should be unaffected by the elbow bend.
    assert _q_deg(result, "left_shoulder_roll_joint") == pytest.approx(90, abs=5)
    assert _q_deg(result, "right_shoulder_roll_joint") == pytest.approx(-90, abs=5)
