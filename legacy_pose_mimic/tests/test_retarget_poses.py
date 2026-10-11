"""Joint-angle checks for the classic poses, in the real G1 convention
(G1_ARM_CONVENTIONS.md section 6). At q = 0 the upper arm hangs and the
forearm points forward, 82 deg apart (measured from the MJCF, not exactly
90). So a STRAIGHT arm has elbow ~ +82 deg and a 90-degree bend has elbow
~ -8 deg. The previous hand-written retargeter's test expected elbow ~ 0
for straight arms, which does not match the G1 model; changed here."""

import math

import numpy as np
import pytest

from pose_fixtures import arm_straight_forward, arms_hanging, t_pose, t_pose_elbows_bent_90

TOL = 12  # degrees; tests/test_mimicry.py checks the resulting directions


def _rest_bend_deg(pipeline):
    rest = pipeline.builder.rest
    return math.degrees(math.acos(float(np.dot(rest.u0["left"], rest.f0["left"]))))


def _solve(pipeline, landmarks):
    pipeline.reset()
    for i in range(3):
        result = pipeline.step(landmarks, i / 30.0)
    return {k: math.degrees(v) for k, v in result.q.items()}


def test_arms_hanging(pipeline):
    q = _solve(pipeline, arms_hanging())
    for side, sign in (("left", 1), ("right", -1)):
        assert abs(q[f"{side}_shoulder_pitch_joint"]) <= TOL
        # Hip clearance keeps the hanging arm ~9 deg out (mediapipe_to_gmr.hip_clearance).
        assert sign * q[f"{side}_shoulder_roll_joint"] == pytest.approx(9, abs=TOL)
        assert q[f"{side}_elbow_joint"] == pytest.approx(_rest_bend_deg(pipeline), abs=TOL)


def test_t_pose_shoulder_roll_and_elbow(pipeline):
    q = _solve(pipeline, t_pose())
    assert q["left_shoulder_roll_joint"] == pytest.approx(90, abs=TOL)
    assert q["right_shoulder_roll_joint"] == pytest.approx(-90, abs=TOL)
    assert q["left_elbow_joint"] == pytest.approx(_rest_bend_deg(pipeline), abs=TOL)
    assert q["right_elbow_joint"] == pytest.approx(_rest_bend_deg(pipeline), abs=TOL)


def test_arm_straight_forward_shoulder_pitch(pipeline):
    q = _solve(pipeline, arm_straight_forward())
    # The G1 shoulder pitch axis is tilted 16 deg, so pitch alone swings the
    # arm inward; roll compensates. Pitch still carries the motion.
    assert q["left_shoulder_pitch_joint"] == pytest.approx(-90, abs=TOL)
    assert q["right_shoulder_pitch_joint"] == pytest.approx(-90, abs=TOL)
    assert q["left_elbow_joint"] == pytest.approx(_rest_bend_deg(pipeline), abs=TOL)


def test_t_pose_elbow_bent_90(pipeline):
    q = _solve(pipeline, t_pose_elbows_bent_90())
    assert q["left_elbow_joint"] == pytest.approx(_rest_bend_deg(pipeline) - 90, abs=TOL)
    assert q["right_elbow_joint"] == pytest.approx(_rest_bend_deg(pipeline) - 90, abs=TOL)
    assert q["left_shoulder_roll_joint"] == pytest.approx(90, abs=TOL)
    assert q["right_shoulder_roll_joint"] == pytest.approx(-90, abs=TOL)
