import math

import pytest

from retarget_upper_body import Retargeter
from pose_fixtures import rotate_about_vertical, t_pose_elbows_bent_90

ONE_DEG = math.radians(1)

ARM_JOINTS = [
    f"{side}_{suffix}_joint"
    for side in ("left", "right")
    for suffix in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
]


def test_same_body_rotated_30_degrees_gives_same_angles():
    body = t_pose_elbows_bent_90()
    rotated = rotate_about_vertical(body, 30)

    result_a = Retargeter().step(body, frame_id=0, dt=1.0)
    result_b = Retargeter().step(rotated, frame_id=0, dt=1.0)

    for joint in ARM_JOINTS:
        a, b = result_a.q[joint], result_b.q[joint]
        assert abs(a - b) <= ONE_DEG, f"{joint}: {math.degrees(a):.2f} vs {math.degrees(b):.2f} deg"
