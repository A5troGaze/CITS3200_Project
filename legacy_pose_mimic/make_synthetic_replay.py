"""Write a synthetic landmark recording (the Step-3 test poses, blended
smoothly one into the next) in the --export-landmarks JSON format, for
exercising replay, the DDS path and --record-demo without a camera.

    python3 make_synthetic_replay.py synthetic.json [--hold 0.6] [--move 0.8] [--fps 30]
"""

import argparse
import json

import numpy as np

from synthetic_poses import base_poses, blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hold", type=float, default=0.6, help="seconds each pose is held")
    parser.add_argument("--move", type=float, default=0.8, help="seconds to blend to the next pose")
    parser.add_argument("--noise", type=float, default=0.004, help="landmark jitter (m), like MediaPipe")
    parser.add_argument("--poses", default=None,
                        help="comma-separated pose names from synthetic_poses.base_poses (default: all)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    poses = base_poses()
    if args.poses:
        by_name = {p.name: p for p in poses}
        poses = [by_name[n] for n in args.poses.split(",")]
    poses = poses + poses[:1]
    frames, k = [], 0
    for a, b in zip(poses, poses[1:]):
        n_hold, n_move = int(args.hold * args.fps), int(args.move * args.fps)
        for i in range(n_hold + n_move):
            s = 0.0 if i < n_hold else (i - n_hold) / n_move
            s = s * s * (3 - 2 * s)
            arr = blend(a, b, s).landmarks
            arr[:, :3] += rng.normal(0, args.noise, size=(33, 3))
            frames.append({
                "frame_id": k,
                "timestamp_ms": int(round(k * 1000 / args.fps)),
                "leader_id": 0,
                "bbox": [0, 0, 0, 0],
                "landmarks_world_m": [
                    {"x": float(r[0]), "y": float(r[1]), "z": float(r[2]), "visibility": float(r[3])} for r in arr
                ],
            })
            k += 1
    with open(args.out, "w") as f:
        json.dump(frames, f)
    print(f"Wrote {len(frames)} frames ({len(frames) / args.fps:.1f} s) to {args.out}")


if __name__ == "__main__":
    main()
