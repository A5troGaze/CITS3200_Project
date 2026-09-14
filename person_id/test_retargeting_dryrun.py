"""Assertion-based unit tests for first-pass arm retargeting."""

import math
import numpy as np

from pose_retargeting import (
    LEFT_ELBOW, LEFT_SHOULDER_PITCH, LEFT_SHOULDER_ROLL,
    RIGHT_ELBOW, RIGHT_SHOULDER_PITCH, RIGHT_SHOULDER_ROLL,
    retarget_arms_indexed,
)


def t_pose():
    xyz = np.zeros((33, 3))
    xyz[11], xyz[12] = [-0.2, 0.5, 0], [0.2, 0.5, 0]
    xyz[13], xyz[14] = [-0.5, 0.5, 0], [0.5, 0.5, 0]
    xyz[15], xyz[16] = [-0.8, 0.5, 0], [0.8, 0.5, 0]
    xyz[23], xyz[24] = [-0.1, 0, 0], [0.1, 0, 0]
    return xyz


def main():
    result = retarget_arms_indexed(t_pose())
    assert abs(result[LEFT_SHOULDER_PITCH]) < 1e-6
    assert abs(result[RIGHT_SHOULDER_PITCH]) < 1e-6
    assert math.isclose(result[LEFT_SHOULDER_ROLL], math.pi / 2, abs_tol=1e-6)
    assert math.isclose(result[RIGHT_SHOULDER_ROLL], -math.pi / 2, abs_tol=1e-6)
    assert abs(result[LEFT_ELBOW]) < 1e-6
    assert abs(result[RIGHT_ELBOW]) < 1e-6
    print("PASS: T-pose produces zero pitch/elbow and mirrored ±90° shoulder roll.")


if __name__ == "__main__":
    main()
