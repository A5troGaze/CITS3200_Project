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


def _hist(peak, bins=128):
    """Synthetic normalised torso histogram with its mass around one bin."""
    h = [0.0] * bins
    for k, w in ((peak - 1, 0.25), (peak, 0.5), (peak + 1, 0.25)):
        h[k % bins] = w
    return h


def _person(cx, hist, cy=240, half_w=60, half_h=160):
    bbox = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    return {"bbox": bbox, "centroid": bbox_centroid(bbox), "hist": hist}


def _crossing(tracker, frames=60, frame_diag=800.0):
    """Leader (red top) walks left->right, another person (blue top)
    right->left at the same speed; their boxes overlap mid-way. Returns
    the leader x positions the tracker reported, frame by frame."""
    red, blue = _hist(3), _hist(80)
    xs = []
    for k in range(frames):
        lx = 100 + k * 440 / (frames - 1)
        ox = 540 - k * 440 / (frames - 1)
        people = [_person(ox, blue), _person(lx, red)]   # other person listed first
        if k == 0:
            tracker.handle_click(lx, 240, [people[1]])
        m = tracker.update(people, frame_diag=frame_diag)
        xs.append((lx, None if m is None else m["centroid"][0]))
    return xs


def test_crossing_people_do_not_swap_identity():
    xs = _crossing(LeaderTracker())
    for true_x, got_x in xs:
        assert got_x is not None
        assert abs(got_x - true_x) < 1e-6, (true_x, got_x)


def test_centroid_only_tracker_does_swap_when_people_cross():
    """Documents the failure the appearance cue fixes: with the old
    behaviour (distance only, no velocity) the lock jumps to the other
    person at the crossing and stays there."""
    xs = _crossing(LeaderTracker(appearance_weight=0.0, max_appearance_distance=1.1, use_velocity=False))
    final_true, final_got = xs[-1]
    assert final_got is not None and abs(final_got - final_true) > 100


def test_match_radius_scales_with_frame_size():
    t = LeaderTracker(max_match_frac=0.2)
    t.handle_click(100, 100, [_detection((50, 50, 150, 150))])
    far = _detection((250, 50, 350, 150))   # 200 px away
    assert t.update([far], frame_diag=800.0) is None    # radius 160 px
    t2 = LeaderTracker(max_match_frac=0.2)
    t2.handle_click(100, 100, [_detection((50, 50, 150, 150))])
    assert t2.update([far], frame_diag=2000.0) is far   # radius 400 px


def test_click_reselects_another_person():
    t = LeaderTracker()
    a, b = _person(100, _hist(3)), _person(500, _hist(80))
    t.handle_click(100, 240, [a, b])
    assert t.update([a, b], frame_diag=800.0) is a
    t.handle_click(500, 240, [a, b])
    assert t.update([a, b], frame_diag=800.0) is b
