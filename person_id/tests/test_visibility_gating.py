from retarget_upper_body import Retargeter
from pose_fixtures import arms_hanging, set_visibility, t_pose, L_ELBOW


def test_low_visibility_side_holds_previous_value_other_side_updates():
    r = Retargeter()

    frame1 = t_pose()
    result1 = r.step(frame1, frame_id=0, dt=1.0)

    frame2 = arms_hanging()
    frame2 = set_visibility(frame2, L_ELBOW, 0.1)  # below the default 0.5 threshold
    result2 = r.step(frame2, frame_id=1, dt=1.0)

    assert result2.updated_sides == {"right"}

    # Left arm: held exactly at frame1's value, not moved toward frame2's
    # (hanging, ~0 rad) target.
    for joint in ("left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                  "left_shoulder_yaw_joint", "left_elbow_joint"):
        assert result2.q[joint] == result1.q[joint]

    # Right arm: updated away from the T-pose value toward the hanging pose.
    assert result2.q["right_shoulder_roll_joint"] != result1.q["right_shoulder_roll_joint"]
