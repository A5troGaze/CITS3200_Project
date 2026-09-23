import json

from retarget_upper_body import Retargeter
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


def test_exported_json_round_trips_through_replay(tmp_path):
    frames = [
        _export_frame(0, 0, arms_hanging()),
        _export_frame(1, 33, t_pose()),
    ]
    export_path = tmp_path / "export.json"
    with open(export_path, "w") as f:
        json.dump(frames, f)

    with open(export_path) as f:
        loaded = json.load(f)

    assert loaded == frames  # exact round-trip through JSON

    # And it actually feeds through the downstream retargeting pipeline
    # (this is what leader_pose.py --replay does per frame).
    retargeter = Retargeter()
    prev_ts = None
    for frame in loaded:
        dt = 1.0 / 30.0 if prev_ts is None else max((frame["timestamp_ms"] - prev_ts) / 1000.0, 1e-3)
        prev_ts = frame["timestamp_ms"]
        result = retargeter.step(frame["landmarks_world_m"], frame["frame_id"], dt)
        assert len(result.q) == 29
        assert result.q_array.shape == (29,)
