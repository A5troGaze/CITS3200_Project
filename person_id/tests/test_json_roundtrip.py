import json

from person_id_replay import load_frames
from pose_fixtures import arms_hanging, t_pose


def _export_frame(frame_id, timestamp_ms, landmarks_world_m):
    """Same schema leader_pose.py's --export-landmarks writes."""
    return {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "leader_id": 0,
        "bbox": [0, 0, 100, 200],
        "landmarks_world_m": landmarks_world_m,
    }


def test_exported_json_round_trips_through_replay(tmp_path, pipeline):
    frames = [
        _export_frame(0, 0, arms_hanging()),
        _export_frame(1, 33, t_pose()),
    ]
    export_path = tmp_path / "export.json"
    with open(export_path, "w") as f:
        json.dump(frames, f)

    with open(export_path) as f:
        assert json.load(f) == frames  # exact round-trip through JSON

    # And it feeds through the replay loader and the pipeline, which is
    # what person_id_replay.py / leader_pose.py --replay do per frame.
    loaded, skipped = load_frames(export_path)
    assert skipped == []
    assert [t for t, _ in loaded] == [0.0, 0.033]
    pipeline.reset()
    for t, landmarks in loaded:
        result = pipeline.step(landmarks, t)
        assert len(result.q) == 29
        assert set(result.targets) == set(range(15, 29))


def test_malformed_frames_are_skipped_not_fatal(tmp_path):
    good = _export_frame(0, 0, t_pose())
    short = _export_frame(1, 33, t_pose()[:20])
    no_ts = {k: v for k, v in _export_frame(2, 66, t_pose()).items() if k != "timestamp_ms"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([good, short, no_ts]))
    loaded, skipped = load_frames(path)
    assert len(loaded) == 1
    assert [fid for fid, _ in skipped] == [1, 2]
