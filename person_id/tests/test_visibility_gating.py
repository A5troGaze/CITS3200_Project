import numpy as np

from pose_fixtures import L_ELBOW, arms_hanging, set_visibility, t_pose


def test_low_visibility_side_holds_previous_value_other_side_updates(pipeline):
    pipeline.reset()
    for i in range(5):
        result1 = pipeline.step(t_pose(), i / 30.0)

    frame2 = set_visibility(arms_hanging(), L_ELBOW, 0.1)  # below the default 0.5 threshold
    for i in range(5, 10):
        result2 = pipeline.step(frame2, i / 30.0)

    assert result2.status["left_upper"] == "hold"
    assert result2.status["left_fore"] == "hold"
    assert result2.status["right_upper"] == "live"

    # Left arm: still out to the side (T-pose), not moved toward hanging.
    seg1, seg2 = result1.info["segments"], result2.info["segments"]
    assert np.allclose(seg2.upper["left"], seg1.upper["left"], atol=1e-9)
    assert abs(result2.q["left_shoulder_roll_joint"] - result1.q["left_shoulder_roll_joint"]) < 0.02

    # Right arm: moved away from the T-pose toward hanging.
    assert result2.q["right_shoulder_roll_joint"] > result1.q["right_shoulder_roll_joint"] + 0.5
