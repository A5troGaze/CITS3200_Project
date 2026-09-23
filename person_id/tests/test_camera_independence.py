import math

from pose_fixtures import rotate_about_vertical, t_pose_elbows_bent_90

ONE_DEG = math.radians(1)

ARM_JOINTS = [
    f"{side}_{suffix}_joint"
    for side in ("left", "right")
    for suffix in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
]


def _solve(pipeline, landmarks):
    pipeline.reset()
    for i in range(3):
        result = pipeline.step(landmarks, i / 30.0)
    return result


def test_same_body_rotated_30_degrees_gives_same_angles(pipeline):
    body = t_pose_elbows_bent_90()
    rotated = rotate_about_vertical(body, 30)

    result_a = _solve(pipeline, body)
    result_b = _solve(pipeline, rotated)

    for joint in ARM_JOINTS:
        a, b = result_a.q[joint], result_b.q[joint]
        assert abs(a - b) <= ONE_DEG, f"{joint}: {math.degrees(a):.2f} vs {math.degrees(b):.2f} deg"
