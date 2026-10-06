from types import SimpleNamespace

import numpy as np

from person_id_brain import PersonIdBrain


class Landmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def person(cx=0.5):
    image = [Landmark(cx + ((i % 3) - 1) * 0.02, 0.4 + ((i % 5) - 2) * 0.02) for i in range(33)]
    world = [Landmark(0.0, 0.0, 0.0) for _ in range(33)]
    return image, world


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.reset_count = 0

    def step(self, landmarks, t):
        self.calls.append((landmarks, t))
        return SimpleNamespace(targets={15: 0.25, 22: -0.25})

    def reset(self):
        self.reset_count += 1


def detector_for(*people):
    pose, world = zip(*people) if people else ([], [])
    result = SimpleNamespace(pose_landmarks=list(pose), pose_world_landmarks=list(world))
    return lambda frame, timestamp_ms: result


def test_single_person_auto_select_returns_brain_output_without_click():
    pipeline = FakePipeline()
    brain = PersonIdBrain(num_people=1, detector=detector_for(person()), pipeline=pipeline)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    output = brain.step(frame, 1.0)

    assert output.joint_targets == {15: 0.25, 22: -0.25}
    assert output.base_velocity is None
    assert brain.locked
    assert brain.last_leader is not None
    assert pipeline.calls[-1][0] is not None


def test_multi_person_requires_owner_selection():
    pipeline = FakePipeline()
    brain = PersonIdBrain(num_people=2, detector=detector_for(person(0.25), person(0.75)), pipeline=pipeline)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert brain.step(frame, 1.0).joint_targets == {}
    assert not brain.locked
    assert pipeline.calls == []

    brain.select_at(0.25 * 640, 0.4 * 480)
    output = brain.step(frame, 1.033)
    assert output.joint_targets
    assert brain.locked


def test_reset_clears_leader_and_pipeline_state_but_keeps_timestamp_monotonic():
    pipeline = FakePipeline()
    brain = PersonIdBrain(num_people=1, detector=detector_for(person()), pipeline=pipeline)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    brain.step(frame, 2.0)
    first_ts = brain.last_timestamp_ms
    assert brain.locked

    brain.reset()
    assert not brain.locked
    assert not brain.engaged
    assert brain.last_output.joint_targets == {}
    assert pipeline.reset_count == 1

    # Gesture mode can switch back within the same synthetic millisecond; keep
    # MediaPipe VIDEO timestamps strictly increasing for the live landmarker.
    brain.step(frame, 2.0)
    assert brain.last_timestamp_ms > first_ts


def test_input_flipped_is_unflipped_before_detection():
    seen = {}

    def detector(frame, timestamp_ms):
        seen["frame"] = frame.copy()
        return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])

    brain = PersonIdBrain(
        num_people=1,
        input_flipped=True,
        detector=detector,
        pipeline=FakePipeline(),
    )
    # Two visibly different columns. gesture_control's incoming frame is the
    # horizontally flipped one; PersonIdBrain should restore raw orientation.
    frame = np.array([[[9, 9, 9], [1, 1, 1]]], dtype=np.uint8)

    output = brain.step(frame, 0.033)

    assert output.joint_targets == {}
    assert seen["frame"].tolist() == [[[1, 1, 1], [9, 9, 9]]]
