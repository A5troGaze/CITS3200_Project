"""Replay an exported landmark recording through the person_id pipeline.

Reads the JSON that `leader_pose.py --export-landmarks` writes: a list of
frames {frame_id, timestamp_ms, leader_id, bbox, landmarks_world_m[33]
{x, y, z, visibility}}. Malformed frames are reported and skipped.

    # Offline: no DDS, no sim. Prints joint targets and GMR latency.
    python3 person_id_replay.py recording.json --dry-run

    # Drive a running unitree_mujoco sim (elastic band on):
    python3 person_id_replay.py recording.json

    # Record the stick figure beside the robot:
    python3 person_id_replay.py recording.json --dry-run --record-demo replay.mp4

The same replay is available as `leader_pose.py --replay recording.json`.
"""

import argparse
import json
import math

import numpy as np


def load_frames(path):
    """Returns (frames, skipped): frames is a list of (t_seconds, (33, 4)
    array), in recording order, keeping only well-formed frames."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("landmark JSON must contain a non-empty list of frames")
    frames, skipped = [], []
    for position, frame in enumerate(raw):
        try:
            t = float(frame["timestamp_ms"]) / 1000.0
            if not math.isfinite(t):
                raise ValueError("timestamp is not finite")
            lms = frame["landmarks_world_m"]
            if not isinstance(lms, list) or len(lms) != 33:
                raise ValueError("expected 33 landmarks")
            arr = np.array([[float(lm["x"]), float(lm["y"]), float(lm["z"]),
                             float(lm.get("visibility", 1.0))] for lm in lms])
            if not np.all(np.isfinite(arr)):
                raise ValueError("non-finite landmark value")
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append((frame.get("frame_id", position) if isinstance(frame, dict) else position, str(exc)))
            continue
        frames.append((t, arr))
    return frames, skipped


def add_pipeline_args(parser):
    """Retargeting options shared by the live and replay entry points."""
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror mode: the person's left arm drives the robot's right arm. "
                             "Default is anatomical (left drives left).")
    parser.add_argument("--waist", choices=("off", "yaw", "3dof"), default="off",
                        help="Default 'off': arms only; the torso is left to the balance controller "
                             "(driving the waist fights the rl_lab / Unitree balance policy). "
                             "'yaw'/'3dof' mimic the torso too, only with a pinned or band-held robot.")
    parser.add_argument("--legs", action="store_true",
                        help="Also command the legs (sim only, with the elastic band on). Off by default.")
    parser.add_argument("--min-visibility", type=float, default=0.5,
                        help="MediaPipe visibility below which a limb is held, then eased to neutral")
    parser.add_argument("--commanded-only", action="store_true",
                        help="Leave joints that are not mimicked limp (kp=kd=0) instead of holding them")
    parser.add_argument("--max-speed", type=float, default=5.0, help="Command velocity cap, rad/s")
    parser.add_argument("--dds-domain", type=int, default=1, help="CycloneDDS domain (sim: 1)")
    parser.add_argument("--dds-interface", default="lo", help="Network interface (sim: lo)")
    parser.add_argument("--real", action="store_true",
                        help="REAL ROBOT via rt/arm_sdk (untested on hardware). Needs --dds-interface "
                             "set to the robot NIC and --dds-domain 0.")


def build_pipeline(args):
    from mimic_pipeline import MimicPipeline

    return MimicPipeline(mirror=args.mirror, waist=args.waist, legs=args.legs,
                         min_visibility=args.min_visibility)


def build_publisher(args, log=print):
    """Sim controller, arm_sdk publisher, or None for --dry-run.
    unitree_sdk2py is only imported here, never in --dry-run."""
    if args.dry_run:
        return None
    if args.real:
        from arm_sdk_publisher import ArmSdkPublisher

        pub = ArmSdkPublisher(interface=args.dds_interface, domain_id=args.dds_domain, real=True, log=log)
        pub.init()
        pub.start()
        pub.engage()
        return pub
    from mujoco_pose_controller import MujocoPoseController

    ctl = MujocoPoseController(args.dds_domain, args.dds_interface, max_command_speed=args.max_speed,
                               commanded_only=args.commanded_only, allow_legs=args.legs)
    log("Waiting for rt/lowstate from unitree_mujoco...")
    ctl.init()
    ctl.start()
    return ctl


def shutdown_publisher(pub):
    """Return the robot to its start pose and stop publishing."""
    if pub is None:
        return
    if hasattr(pub, "disengage"):
        pub.disengage()
    else:
        if pub.ready.is_set():
            pub.return_to_start()
    pub.stop()


def report_tracking(track, log=print, settle_frames=20, settle_tol=0.02, lags=range(0, 16)):
    """Compare the joint positions read back from rt/lowstate with the
    targets sent (arms + waist), two ways:

    * settled: frames where that joint's target has moved less than
      settle_tol rad over the previous settle_frames frames (the person is
      holding a pose), i.e. steady-state tracking accuracy;
    * lag: the delay (in frames) that best lines the measured motion up
      with the targets, i.e. how far the robot trails the person.
    Returns (settled median deg, settled p95 deg, lag frames)."""
    idx = sorted(track[0][0])
    targets = np.array([[tg[i] for i in idx] for tg, _ in track])
    measured = np.array([[m[i] for i in idx] for _, m in track])
    n = len(targets)
    errs = []
    for k in range(settle_frames, n):
        window = targets[k - settle_frames:k + 1]
        steady = (window.max(axis=0) - window.min(axis=0)) < settle_tol
        errs.extend(np.abs(measured[k, steady] - targets[k, steady]))
    errs = np.degrees(np.array(errs)) if errs else np.array([np.nan])
    lag_err = []
    for lag in lags:
        e = np.abs(measured[lag:] - targets[:n - lag]) if lag else np.abs(measured - targets)
        lag_err.append(np.median(e))
    best_lag = int(np.argmin(lag_err))
    log("\nTracking (rt/lowstate vs targets sent), arms + waist:")
    log(f"  settled frames: median {np.median(errs):.2f} deg, p95 {np.percentile(errs, 95):.2f} deg, "
        f"max {np.max(errs):.2f} deg ({len(errs)} joint-samples)")
    log(f"  while moving the robot trails the targets by ~{best_lag} frames (~{best_lag * 33} ms)")
    return float(np.median(errs)), float(np.percentile(errs, 95)), best_lag


def run_replay(args, log=print):
    """Replay args.landmarks_json through the pipeline. args needs
    landmarks_json, dry_run, record_demo, no_realtime and the
    add_pipeline_args options."""
    import time

    frames, skipped = load_frames(args.landmarks_json)
    for frame_id, reason in skipped:
        log(f"Frame {frame_id} skipped: {reason}")
    if not frames:
        raise SystemExit("No valid frames to replay")

    pipeline = build_pipeline(args)
    recorder = None
    if args.record_demo:
        from demo_recorder import DemoRecorder

        fps = 1.0 / max(np.median(np.diff([t for t, _ in frames])), 1e-3) if len(frames) > 1 else 30.0
        recorder = DemoRecorder(args.record_demo, fps=fps, log=log)

    pub = None
    try:
        pub = build_publisher(args, log=log)
        start = time.monotonic()
        t0 = frames[0][0]
        lat, lat_cpu = [], []
        track = []   # (target, measured) per frame, for --report-tracking
        for k, (t, arr) in enumerate(frames):
            if not args.no_realtime and pub is not None:
                time.sleep(max(0.0, (t - t0) - (time.monotonic() - start)))
            result = pipeline.step(arr, t)
            lat.append(result.timings_ms["gmr"])
            lat_cpu.append(result.timings_ms["gmr_cpu"])
            if pub is not None:
                pub.set_targets(result.targets)
                if getattr(args, "report_tracking", False) and hasattr(pub, "measured_positions"):
                    track.append((dict(result.targets), pub.measured_positions()))
            if args.dry_run and k % 10 == 0:
                log(f"t={t - t0:6.2f}s gmr {result.timings_ms['gmr']:5.1f} ms "
                    + " ".join(f"{i}:{q:+.2f}" for i, q in sorted(result.targets.items())))
            if recorder is not None:
                recorder.add_stick_frame(arr, result)
        log(f"Replay complete: {len(frames)} frames, {len(skipped)} skipped. "
            f"GMR per frame: wall median {np.median(lat):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms; "
            f"CPU median {np.median(lat_cpu):.1f} ms, p95 {np.percentile(lat_cpu, 95):.1f} ms.")
        if track:
            report_tracking(track, log)
    except KeyboardInterrupt:
        log("Interrupted.")
    except TimeoutError as exc:
        raise SystemExit(f"Could not connect: {exc}")
    finally:
        shutdown_publisher(pub)
        if recorder is not None:
            recorder.close()
            log(f"Demo video written to {args.record_demo}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("landmarks_json")
    parser.add_argument("--dry-run", action="store_true", help="No DDS: print targets only")
    parser.add_argument("--record-demo", default=None, help="Write stick figure | robot video (mp4)")
    parser.add_argument("--no-realtime", action="store_true", help="Do not wait between frames")
    parser.add_argument("--report-tracking", action="store_true",
                        help="Sim only: compare rt/lowstate joint positions with the targets sent")
    add_pipeline_args(parser)
    run_replay(parser.parse_args())


if __name__ == "__main__":
    main()
