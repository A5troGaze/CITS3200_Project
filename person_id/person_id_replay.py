"""Replay exported leader landmarks through retargeting and G1 MuJoCo.

The input is a JSON list of frames containing ``timestamp_ms`` and
``landmarks_world_m``. Each landmark frame must contain exactly 33 entries with
numeric x, y, and z values. Valid frames are replayed using their recorded timing;
malformed frames are reported and skipped.
"""

import argparse
import json
import math
import time

from mujoco_pose_controller import MujocoPoseController
from pose_retargeting import json_landmarks_to_xyz, retarget_arms_indexed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("landmarks_json")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        with open(args.landmarks_json, encoding="utf-8") as handle:
            frames = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read landmark JSON: {exc}")
    if not isinstance(frames, list) or not frames:
        raise SystemExit("Landmark JSON must contain a non-empty frame list")

    controller = MujocoPoseController(args.domain_id, args.interface)
    try:
        controller.init()
        controller.start()
        previous_timestamp = None  # Updated only after successfully parsing a frame.
        skipped = 0
        for position, frame in enumerate(frames):
            try:
                timestamp = float(frame["timestamp_ms"])
                if not math.isfinite(timestamp):
                    raise ValueError("timestamp is not finite")
                xyz = json_landmarks_to_xyz(frame.get("landmarks_world_m"))
                targets = retarget_arms_indexed(xyz)
            except (KeyError, TypeError, ValueError) as exc:
                # One malformed frame should not abort the remainder of a replay.
                skipped += 1
                print(f"Frame {frame.get('frame_id', position)} skipped: {exc}")
                continue

            if previous_timestamp is not None:
                # Preserve the elapsed time between valid frames in the recording.
                time.sleep(max(0.0, timestamp - previous_timestamp) / 1000.0)
            controller.set_targets(targets)
            previous_timestamp = timestamp
        print(f"Replay complete; {skipped} malformed frame(s) skipped.")
    except TimeoutError as exc:
        raise SystemExit(f"Could not connect to MuJoCo: {exc}")
    finally:
        if controller.ready.is_set():
            controller.return_to_start()
        controller.stop()


if __name__ == "__main__":
    main()
