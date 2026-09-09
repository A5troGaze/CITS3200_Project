"""
Leader Pose — canonical person_id capture script
CITS3200 Humanoid Project | Person Identification subgroup

Replaces pose_test.py, pose_test-world_landmarks.py and
pose_test-multi-world-landmarks.py (deleted in this same commit). All
shared logic now lives in pose_common.py; this file is the CLI + main loop.

What this does:
  - Reads a live camera feed or a video file, OR replays a previously
    recorded --export-landmarks JSON file with no camera at all
    (--replay), for testing the downstream retargeting/MuJoCo pipeline in
    the VM without a webcam.
  - Detects multiple people per frame (--num-people). With --num-people 1
    the sole detection is auto-selected as leader with no click needed
    (matching the old single-person script's behaviour); with more than
    one, click a person in the preview window to select them as leader —
    click again anytime to switch. If the leader is briefly lost, tracking
    holds for --leader-lost-frames frames before requiring a re-click.
  - Exports per-frame LEADER-ONLY MediaPipe world landmarks as JSON
    (metres, hip-centred): {frame_id, timestamp_ms, leader_id, bbox,
    landmarks_world_m[33]{x,y,z,visibility}} — the schema
    pose_test-multi-world-landmarks.py established; --replay reads this
    same schema back in.
  - --mujoco retargets the leader's world landmarks to G1 arm joint
    targets (retarget_upper_body.py) and streams them to a running
    Unitree MuJoCo sim over DDS (mujoco_link.py).
  - --dry-run runs that same retargeting step and prints the 29 joint
    targets per frame instead of publishing anything — no DDS, no
    unitree_sdk2py import, no sim required. Works with --replay for fully
    offline testing, or live/video-file input.

Before running:
  1. source ~/CITS3200/Dependencies/g1-env/bin/activate
  2. Model file expected at the path resolve_model_path() picks (see
     pose_common.py): --model overrides, else $CITS3200_MODELS_DIR, else
     ~/CITS3200/Dependencies/Models/pose_landmarker.task.

Usage:
  # Live webcam, multi-person, click a person to select them:
  python3 leader_pose.py --num-people 4

  # Live webcam, single person, auto-selected, headless:
  python3 leader_pose.py --num-people 1 --no-preview

  # Also write annotated output + leader-only world-landmark export:
  python3 leader_pose.py --output out.mp4 --export-landmarks out.json

  # From a video file instead of webcam:
  python3 leader_pose.py --input test_clip.mp4

  # Replay a recording through the retargeter with no camera, no DDS:
  python3 leader_pose.py --replay out.json --dry-run

  # Replay through the retargeter and stream to a running MuJoCo sim:
  python3 leader_pose.py --replay out.json --mujoco --dds-domain 1 --dds-interface lo

  # Live webcam straight into the sim:
  python3 leader_pose.py --mujoco
"""

import argparse
import json
import time

from pose_common import (
    FpsCounter,
    LeaderTracker,
    bbox_centroid,
    build_landmarker,
    draw_skeleton,
    landmarks_to_bbox,
    make_timestamp,
    open_capture,
    open_video_writer,
    resolve_model_path,
)

WINDOW_NAME = "Leader Pose - click a person to select leader, 'q' to quit"


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-person leader detection, tracking and G1 retargeting")
    parser.add_argument("--input", default="0", help="Camera index (default 0) or path to a video file")
    parser.add_argument("--output", default=None, help="Optional path to write annotated video")
    parser.add_argument("--model", default=None, help="Path to the .task pose landmarker model (see resolve_model_path)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (lower = faster, less laggy)")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--export-landmarks", default=None,
                         help="Optional path to dump per-frame LEADER-ONLY world landmarks as JSON")
    parser.add_argument("--no-preview", action="store_true",
                         help="No preview window. Only valid with --num-people 1 (no click needed to select leader) or --replay")
    parser.add_argument("--num-people", type=int, default=4,
                         help="Max number of people to detect per frame. 1 auto-selects the sole detection with no click")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-match-distance", type=float, default=120.0,
                         help="Max pixel distance between frames for the leader's bbox centroid to still match")
    parser.add_argument("--leader-lost-frames", type=int, default=15,
                         help="Consecutive unmatched frames tolerated before the leader lock is released")
    parser.add_argument("--replay", default=None,
                         help="Path to a previously exported landmarks JSON file. Feeds it through the "
                              "downstream retargeting/MuJoCo pipeline with no camera at all.")
    parser.add_argument("--mujoco", action="store_true",
                         help="Retarget the leader's world landmarks and stream them to a running "
                              "Unitree MuJoCo sim over DDS (see mujoco_link.py)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run retargeting and print the 29 joint targets per frame; no DDS, no sim required")
    parser.add_argument("--min-visibility", type=float, default=0.5,
                         help="Per-side arm-joint hold threshold: if shoulder/elbow/wrist visibility on a "
                              "side drops below this, that side's arm joints hold their previous value")
    parser.add_argument("--dds-domain", type=int, default=1, help="CycloneDDS domain ID for the MuJoCo bridge")
    parser.add_argument("--dds-interface", default="lo", help="Network interface for the MuJoCo bridge")
    return parser.parse_args()


def _build_retargeter(args):
    from retarget_upper_body import Retargeter
    return Retargeter(min_visibility=args.min_visibility)


def _build_mujoco_link(args):
    from mujoco_link import MujocoLink
    return MujocoLink(domain_id=args.dds_domain, interface=args.dds_interface)


def run_replay(args):
    if not (args.mujoco or args.dry_run):
        raise SystemExit("--replay only makes sense with --mujoco and/or --dry-run (otherwise there is "
                          "nothing downstream to feed it to)")

    with open(args.replay) as f:
        frames = json.load(f)

    retargeter = _build_retargeter(args)
    mujoco_link = _build_mujoco_link(args) if (args.mujoco and not args.dry_run) else None

    prev_timestamp_ms = None
    try:
        for frame in frames:
            timestamp_ms = frame["timestamp_ms"]
            dt = 1.0 / 30.0 if prev_timestamp_ms is None else max((timestamp_ms - prev_timestamp_ms) / 1000.0, 1e-3)
            prev_timestamp_ms = timestamp_ms

            lowstate_q = mujoco_link.latest_lowstate_q() if mujoco_link is not None else None
            result = retargeter.step(frame["landmarks_world_m"], frame["frame_id"], dt, lowstate_q)

            if args.dry_run:
                print(f"frame {result.frame_id} (updated={sorted(result.updated_sides)}): {result.q_array.tolist()}")
            if mujoco_link is not None:
                mujoco_link.publish_target(result)
    finally:
        if mujoco_link is not None:
            mujoco_link.close()

    print(f"Replay done. Processed {len(frames)} frames from '{args.replay}'.")


def run_live(args):
    import cv2
    import mediapipe as mp

    if args.no_preview and args.num_people != 1:
        raise SystemExit("--no-preview requires --num-people 1 (manual leader selection needs the preview "
                          "window to click on; with one person there is nothing to click)")

    model_path = resolve_model_path(args.model)
    cap, frame_w, frame_h, fps, is_live_camera = open_capture(args.input, args.width, args.height)
    writer = open_video_writer(args.output, fps, frame_w, frame_h)
    landmarker = build_landmarker(
        model_path, args.num_people,
        args.min_detection_confidence, args.min_presence_confidence, args.min_tracking_confidence,
    )

    retargeter = _build_retargeter(args) if (args.mujoco or args.dry_run) else None
    mujoco_link = _build_mujoco_link(args) if (args.mujoco and not args.dry_run) else None

    click_state = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["xy"] = (x, y)

    if not args.no_preview:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    tracker = LeaderTracker(max_match_distance=args.max_match_distance, leader_lost_frames=args.leader_lost_frames)
    fps_counter = FpsCounter()

    all_frame_landmarks = []
    frame_index = 0
    start_time = time.time()
    prev_timestamp_ms = None

    print(f"Running on '{args.input}' at {frame_w}x{frame_h}, detecting up to {args.num_people} people.")
    if args.num_people == 1:
        print("Single-person mode: the sole detection auto-selects as leader.")
    elif not args.no_preview:
        print("Click a person in the preview window to select them as the leader. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = make_timestamp(is_live_camera, start_time, frame_index, fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            detections = []
            world_results = result.pose_world_landmarks
            for pose_index, landmarks in enumerate(result.pose_landmarks):
                world_landmarks = world_results[pose_index] if pose_index < len(world_results) else None
                bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)
                detections.append({
                    "landmarks": landmarks,
                    "world_landmarks": world_landmarks,
                    "bbox": bbox,
                    "centroid": bbox_centroid(bbox),
                })

            if args.num_people == 1:
                if not tracker.locked and detections:
                    tracker.handle_click(*detections[0]["centroid"], detections)
            elif click_state["xy"] is not None:
                cx, cy = click_state["xy"]
                click_state["xy"] = None
                tracker.handle_click(cx, cy, detections)

            leader_detection = tracker.update(detections)

            if not args.no_preview:
                for d in detections:
                    x_min, y_min, x_max, y_max = d["bbox"]
                    if leader_detection is not None and d is leader_detection:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 215, 255), 3)
                        cv2.putText(frame, "Leader", (x_min, max(y_min - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
                        draw_skeleton(frame, d["landmarks"], frame_w, frame_h, color=(0, 200, 0))
                    else:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (90, 90, 90), 1)

                if tracker.locked and leader_detection is None:
                    cv2.putText(frame, "Leader lost - hold on...", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif not tracker.locked and args.num_people > 1:
                    cv2.putText(frame, "Click a person to select leader", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            world_landmarks = leader_detection["world_landmarks"] if leader_detection is not None else None
            if world_landmarks is not None:
                x_min, y_min, x_max, y_max = leader_detection["bbox"]
                landmarks_world_m = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in world_landmarks
                ]

                if args.export_landmarks:
                    all_frame_landmarks.append({
                        "frame_id": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "leader_id": 0,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "landmarks_world_m": landmarks_world_m,
                    })

                if retargeter is not None:
                    dt = 1.0 / fps if prev_timestamp_ms is None else max((timestamp_ms - prev_timestamp_ms) / 1000.0, 1e-3)
                    prev_timestamp_ms = timestamp_ms
                    lowstate_q = mujoco_link.latest_lowstate_q() if mujoco_link is not None else None
                    retarget_result = retargeter.step(landmarks_world_m, frame_index, dt, lowstate_q)
                    if args.dry_run:
                        print(f"frame {frame_index} (updated={sorted(retarget_result.updated_sides)}): "
                              f"{retarget_result.q_array.tolist()}")
                    if mujoco_link is not None:
                        mujoco_link.publish_target(retarget_result)

            if not args.no_preview:
                display_fps = fps_counter.tick()
                cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            if not args.no_preview:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    except KeyboardInterrupt:
        # Caught here (not left to propagate) so cleanup still runs AND the
        # summary below still prints.
        print("\nInterrupted by user (Ctrl-C).")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        landmarker.close()
        if mujoco_link is not None:
            mujoco_link.close()

    elapsed = time.time() - start_time
    avg_fps = frame_index / elapsed if elapsed > 0 else 0.0
    print(f"Done. Processed {frame_index} frames in {elapsed:.1f}s ({avg_fps:.2f} fps average).")
    if args.export_landmarks:
        with open(args.export_landmarks, "w") as f:
            json.dump(all_frame_landmarks, f, indent=2)
        print(f"Leader-only world landmarks exported to {args.export_landmarks}")


def main():
    args = parse_args()
    if args.replay:
        run_replay(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
