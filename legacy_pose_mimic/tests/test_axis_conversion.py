import numpy as np
import pytest

from mediapipe_to_gmr import mediapipe_to_robot


def test_point_toward_camera_maps_to_robot_forward():
    # z toward the camera is negative in MediaPipe's world-landmark frame.
    robot = mediapipe_to_robot(np.array([0.0, 0.0, -1.0]))
    assert robot == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_point_below_hips_maps_to_robot_down():
    # y is down in MediaPipe's frame.
    robot = mediapipe_to_robot(np.array([0.0, 1.0, 0.0]))
    assert robot == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_point_to_image_right_maps_to_robot_left():
    robot = mediapipe_to_robot(np.array([1.0, 0.0, 0.0]))
    assert robot == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_batched_points_same_as_single():
    points = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    batched = mediapipe_to_robot(points)
    for i, p in enumerate(points):
        assert batched[i] == pytest.approx(mediapipe_to_robot(p), abs=1e-9)
