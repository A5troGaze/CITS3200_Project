from pose_common import LeaderTracker, bbox_centroid


def _detection(bbox):
    return {"bbox": bbox, "centroid": bbox_centroid(bbox)}


def test_leader_released_after_lost_frames_plus_one_empty_frames():
    tracker = LeaderTracker(max_match_distance=50.0, leader_lost_frames=3)
    tracker.handle_click(50, 50, [_detection((0, 0, 100, 100))])
    assert tracker.locked

    # Regression test for defect 1: update() must be called (and must
    # advance lost_frames) even on completely empty frames.
    for _ in range(3):
        tracker.update([])
        assert tracker.locked, "should still be holding within the grace period"

    tracker.update([])  # the (leader_lost_frames + 1)th empty frame
    assert not tracker.locked


def test_click_inside_overlapping_boxes_picks_the_smaller():
    tracker = LeaderTracker()
    big = _detection((0, 0, 200, 200))
    small = _detection((40, 40, 60, 60))
    selected = tracker.handle_click(50, 50, [big, small])
    assert selected is small


def test_detection_farther_than_max_match_distance_does_not_steal_lock():
    tracker = LeaderTracker(max_match_distance=10.0, leader_lost_frames=5)
    near = _detection((0, 0, 20, 20))       # centroid (10, 10)
    tracker.handle_click(10, 10, [near])
    assert tracker.locked

    far = _detection((500, 500, 520, 520))  # centroid (510, 510), far away
    matched = tracker.update([far])
    assert matched is None
    assert tracker.locked  # still locked, just unmatched this frame (grace period)
    assert tracker.lost_frames == 1
