"""
Leader Pose: the person_id live entry point.
CITS3200 Humanoid Project | Person Identification subgroup

Camera (or video file) -> MediaPipe multi-person pose -> click-selected
leader, tracked by position + clothing colour -> MimicPipeline (one-euro
filter, visibility gate, GMR retargeting) -> G1 joint targets -> the
unitree_mujoco sim over rt/lowcmd (or, with --real, the robot over
rt/arm_sdk; untested on hardware).

    # Live camera into the running sim (start unitree_mujoco first, band on):
    python3 leader_pose.py

    # No sim, no DDS (unitree_sdk2py is not even imported):
    python3 leader_pose.py --dry-run

    # Record the client proof video (camera | simulated G1), composed on exit:
    python3 leader_pose.py --record-demo demo.mp4

    # Save the leader's landmarks for later replay:
    python3 leader_pose.py --dry-run --export-landmarks session.json
    python3 leader_pose.py --replay session.json            # into the sim
    python3 leader_pose.py --replay session.json --dry-run  # offline

Keys in the preview window: click a person to (re)select the leader,
'q' or Esc to quit. With --num-people 1 the only person is selected
automatically. Ctrl-C, 'q' or losing the camera all return the robot's
arms to where they started and stop publishing.

See RUNNING.md for the full set of commands and troubleshooting.
"""

import argparse
import json
import math
import time

from person_id_replay import add_pipeline_args, build_pipeline, build_publisher, run_replay, shutdown_publisher
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
    torso_histogram,
)

WINDOW_NAME = "Leader Pose - click a person to select leader, 'q' to quit"
STATUS_COLOURS = {"live": (0, 200, 0), "hold": (0, 215, 255), "ease": (0, 140, 255), "neutral": (120, 120, 120)}
CAMERA_LOSS_FRAMES = 30   # consecutive failed reads before giving up on a live camera


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("input")
    src.add_argument("--input", default="0", help="Camera index (default 0) or path to a video file")
    src.add_argument("--replay", default=None, metavar="JSON",
                     help="Replay an --export-landmarks file instead of using a camera")
    src.add_argument("--model", default=None, help="Path to pose_landmarker.task (see pose_common.resolve_model_path)")
    src.add_argument("--width", type=int, default=640)
    src.add_argument("--height", type=int, default=480)
    det = parser.add_argument_group("detection and leader tracking")
    det.add_argument("--num-people", type=int, default=4,
                     help="Max people per frame. 1 selects the only person automatically")
    det.add_argument("--min-detection-confidence", type=float, default=0.5)
    det.add_argument("--min-presence-confidence", type=float, default=0.5)
    det.add_argument("--min-tracking-confidence", type=float, default=0.5)
    det.add_argument("--max-match-frac", type=float, default=0.2,
                     help="Leader match radius as a fraction of the frame diagonal")
    det.add_argument("--leader-lost-frames", type=int, default=15,
                     help="Unmatched frames before the leader lock is released")
    out = parser.add_argument_group("output")
    out.add_argument("--dry-run", action="store_true",
                     help="No DDS and no robot: run detection and retargeting only")
    out.add_argument("--mujoco", action="store_true", help=argparse.SUPPRESS)  # old flag; sim is now the default
    out.add_argument("--output", default=None, help="Write the annotated camera view to this video file")
    out.add_argument("--record-demo", default=None, metavar="MP4",
                     help="Write camera | simulated G1 side by side (rendered after the session ends)")
    out.add_argument("--export-landmarks", default=None, metavar="JSON",
                     help="Save the leader's world landmarks each frame, for --replay")
    out.add_argument("--no-preview", action="store_true",
                     help="No window (only with --num-people 1, since selecting a leader needs a click)")
    out.add_argument("--no-realtime", action="store_true", help="--replay: do not wait between frames")
    add_pipeline_args(parser)
    return parser.parse_args(argv)


def _draw_status(cv2, frame, result, mirror, fps):
    y = 25
    cv2.putText(frame, f"FPS {fps:4.1f}  {'MIRROR' if mirror else 'anatomical'}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if result is None:
        return
    y += 22
    cv2.putText(frame, f"GMR {result.timings_ms['gmr']:4.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    x = 10
    for name in ("torso", "left_upper", "left_fore", "right_upper", "right_fore"):
        status = result.status[name]
        y2 = y + 20
        label = f"{name}:{status}"
        cv2.putText(frame, label, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, STATUS_COLOURS[status], 1)
        x += 9 * len(label) + 6


def run_live(args):
    import cv2
    import mediapipe as mp

    if args.no_preview and args.num_people != 1:
        raise SystemExit("--no-preview needs --num-people 1: selecting a leader needs a click in the window")

    model_path = resolve_model_path(args.model)
    cap, frame_w, frame_h, fps, is_live_camera = open_capture(args.input, args.width, args.height)
    # Some cameras (the VirtualBox webcam passthrough) ignore the requested
    # size and send 1280x720. Downscale to the requested width so detection,
    # drawing and recording don't pay for pixels nobody needs.
    scale = min(1.0, args.width / frame_w)
    if scale < 1.0:
        frame_w, frame_h = int(round(frame_w * scale)), int(round(frame_h * scale))
        print(f"Downscaling camera frames to {frame_w}x{frame_h}.")
    frame_diag = math.hypot(frame_w, frame_h)
    writer = open_video_writer(args.output, fps, frame_w, frame_h)
    landmarker = build_landmarker(model_path, args.num_people, args.min_detection_confidence,
                                  args.min_presence_confidence, args.min_tracking_confidence)
    pipeline = build_pipeline(args)
    recorder = None
    if args.record_demo:
        from demo_recorder import DemoRecorder

        recorder = DemoRecorder(args.record_demo, fps=fps)

    click = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["xy"] = (x, y)

    if not args.no_preview:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    tracker = LeaderTracker(leader_lost_frames=args.leader_lost_frames, max_match_frac=args.max_match_frac)
    fps_counter = FpsCounter()
    exported = []
    frame_index = 0
    failed_reads = 0
    engaged = False
    start_time = time.time()
    pub = None
    stop_reason = "end of input"
    try:
        pub = build_publisher(args)
        print(f"Running on '{args.input}' at {frame_w}x{frame_h}; "
              f"{'dry run (no DDS)' if pub is None else 'publishing'}; "
              f"{'mirror' if args.mirror else 'anatomical'} mapping.")
        if args.num_people > 1 and not args.no_preview:
            print("Click a person in the window to select the leader. 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_live_camera:
                    failed_reads += 1
                    if failed_reads >= CAMERA_LOSS_FRAMES:
                        stop_reason = "camera lost"
                        break
                    time.sleep(0.01)
                    continue
                break
            failed_reads = 0
            if scale < 1.0:
                frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = make_timestamp(is_live_camera, start_time, frame_index, fps)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

            detections = []
            worlds = result.pose_world_landmarks
            for i, landmarks in enumerate(result.pose_landmarks):
                bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)
                detections.append({
                    "landmarks": landmarks,
                    "world_landmarks": worlds[i] if i < len(worlds) else None,
                    "bbox": bbox,
                    "centroid": bbox_centroid(bbox),
                    "hist": torso_histogram(frame, landmarks, frame_w, frame_h) if args.num_people > 1 else None,
                })

            if args.num_people == 1:
                if not tracker.locked and detections:
                    tracker.handle_click(*detections[0]["centroid"], detections)
            elif click["xy"] is not None:
                tracker.handle_click(*click["xy"], detections)
                click["xy"] = None
            leader = tracker.update(detections, frame_diag=frame_diag)

            world = leader["world_landmarks"] if leader is not None else None
            landmarks_world_m = None
            if world is not None:
                landmarks_world_m = [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in world]
                if args.export_landmarks:
                    exported.append({"frame_id": frame_index, "timestamp_ms": timestamp_ms, "leader_id": 0,
                                     "bbox": list(leader["bbox"]), "landmarks_world_m": landmarks_world_m})

            # Once a leader has been selected the pipeline runs every frame:
            # with no leader it holds the last pose, then eases to neutral.
            engaged = engaged or tracker.locked
            mimic = pipeline.step(landmarks_world_m, timestamp_ms / 1000.0) if engaged else None
            if mimic is not None and pub is not None:
                pub.set_targets(mimic.targets)
            if args.dry_run and mimic is not None and frame_index % 15 == 0:
                print(f"frame {frame_index}: gmr {mimic.timings_ms['gmr']:.1f} ms "
                      + " ".join(f"{i}:{q:+.2f}" for i, q in sorted(mimic.targets.items())))

            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                if d is leader:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 3)
                    cv2.putText(frame, "Leader", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 215, 255), 2)
                    draw_skeleton(frame, d["landmarks"], frame_w, frame_h)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
            if tracker.locked and leader is None:
                cv2.putText(frame, "Leader lost - holding", (10, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
            elif not tracker.locked and args.num_people > 1:
                cv2.putText(frame, "Click a person to select the leader", (10, frame_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            _draw_status(cv2, frame, mimic, args.mirror, fps_counter.tick())

            if writer is not None:
                writer.write(frame)
            if recorder is not None:
                recorder.add_camera_frame(frame, mimic)
            if not args.no_preview:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    stop_reason = "quit"
                    break
            frame_index += 1
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    except TimeoutError as exc:
        raise SystemExit(f"Could not connect to the simulator: {exc}")
    finally:
        print(f"Stopping ({stop_reason}): returning the robot to its start pose.")
        shutdown_publisher(pub)
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        landmarker.close()
        if args.export_landmarks:
            with open(args.export_landmarks, "w") as f:
                json.dump(exported, f)
            print(f"Leader landmarks for {len(exported)} frames written to {args.export_landmarks}")
        if recorder is not None:
            recorder.close()
            print(f"Demo video written to {args.record_demo}")

    elapsed = time.time() - start_time
    print(f"Done: {frame_index} frames in {elapsed:.1f} s ({frame_index / max(elapsed, 1e-9):.1f} fps).")


def main(argv=None):
    args = parse_args(argv)
    if args.replay:
        args.landmarks_json = args.replay
        run_replay(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
