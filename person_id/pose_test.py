"""
Multi-Person Detection + Leader Selection/Tracking
CITS3200 Humanoid Project | Person Identification subgroup

What this does:
  - Reads a live camera feed (or a video file)
  - Detects MULTIPLE people per frame using MediaPipe Pose Landmarker
  - Lets you click on one person in the preview window to designate them
    the "leader" — tracking then follows that specific person frame-to-
    frame (via nearest-centroid matching), while every other detected
    person is drawn dimmed and excluded from the exported data entirely.
  - Click again at any time to switch the leader to someone else.
  - If the leader briefly leaves frame / is occluded, tracking holds for
    a grace period (--leader-lost-frames) before requiring a re-click.
  - Shows a live FPS counter so lag is visible while testing, not guessed at.
  - Optionally writes the annotated stream to a video file.
  - Optionally exports per-frame LEADER-ONLY landmark data as JSON, using
    the hand-off schema the kinematics/gesture team needs:
    {frame_id, bbox, landmarks[33], timestamp}

Before running:
  1. source ~/CITS3200/Dependencies/g1-env/bin/activate
  2. Model file expected at --model path below (already downloaded per
     team setup). Override with --model /path/to/pose_landmarker.task
     if yours lives elsewhere.

Usage:
  # Live webcam, click a person in the preview window to select them:
  python3 pose_test.py

  # Detect up to 6 people instead of the default 4:
  python3 pose_test.py --num-people 6

  # Live webcam, also write annotated output + leader-only landmarks:
  python3 pose_test.py --output out.mp4 --export-landmarks out.json

  # From a video file instead of webcam:
  python3 pose_test.py --input test_clip.mp4 --output out.mp4

  # If lag is an issue, drop resolution further:
  python3 pose_test.py --width 480 --height 360

  # Loosen detection if people aren't being picked up in poor lighting:
  python3 pose_test.py --min-detection-confidence 0.3

Note: manual leader selection requires the preview window (it's how you
click), so this will refuse to run with --no-preview.
"""

import argparse
import json
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL_PATH = "/home/ubuntu/CITS3200/models/pose_landmarker.task"
WINDOW_NAME = "Person Identification - click a person to select leader, 'q' to quit"

# MediaPipe's documented skeletal connections for the 33 pose landmarks —
# drawing these (not just dots) is what makes the overlay actually read as
# a skeleton in the demo, matching what the client asked for.
# Hardcoded here (rather than pulled from mp.solutions.pose) because
# Tasks-API-only MediaPipe installs don't always expose the legacy
# `solutions` submodule. This is a fixed fact about the model's output
# structure, not something that should ever be a config option.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def build_landmarker(model_path, num_people, min_detection_confidence,
                      min_presence_confidence, min_tracking_confidence):
    """Multi-person MediaPipe PoseLandmarker in VIDEO mode."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_people,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def landmarks_to_bbox(landmarks, frame_w, frame_h, padding=20):
    """Turn normalized landmarks into a pixel-space bounding box, clamped to frame edges."""
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    x_min = max(int(min(xs)) - padding, 0)
    y_min = max(int(min(ys)) - padding, 0)
    x_max = min(int(max(xs)) + padding, frame_w)
    y_max = min(int(max(ys)) + padding, frame_h)
    return x_min, y_min, x_max, y_max


def bbox_centroid(bbox):
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def draw_skeleton(frame, landmarks, frame_w, frame_h, color=(0, 200, 0)):
    """Draw connected skeleton lines plus joint dots."""
    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 3, color, -1)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-person detection with click-to-select leader tracking")
    parser.add_argument("--input", default="0", help="Camera index (default 0) or path to a video file")
    parser.add_argument("--output", default=None, help="Optional path to write annotated video")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the .task pose landmarker model")
    parser.add_argument("--width", type=int, default=640, help="Capture width (lower = faster, less laggy)")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--export-landmarks", default=None,
                         help="Optional path to dump per-frame LEADER-ONLY landmarks as JSON")
    parser.add_argument("--no-preview", action="store_true",
                         help="Not supported: manual leader selection needs the preview window to click on")
    parser.add_argument("--num-people", type=int, default=4,
                         help="Max number of people to detect per frame (was hardcoded to 4, now configurable)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5,
                         help="MediaPipe min_pose_detection_confidence (lower = more detections, more false positives)")
    parser.add_argument("--min-presence-confidence", type=float, default=0.5,
                         help="MediaPipe min_pose_presence_confidence")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5,
                         help="MediaPipe min_tracking_confidence")
    parser.add_argument("--max-match-distance", type=float, default=120.0,
                         help="Max pixel distance between frames for the leader's bounding-box centroid to "
                              "still count as the same person (tune down if it jumps to the wrong person "
                              "in a crowded scene, up if fast movement loses tracking)")
    parser.add_argument("--leader-lost-frames", type=int, default=15,
                         help="How many consecutive frames the leader can go unmatched (occlusion, briefly "
                              "out of frame) before tracking gives up and requires a re-click")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_preview:
        raise SystemExit(
            "Manual leader selection requires the preview window, since clicking on a "
            "person in that window is how you select them. Remove --no-preview to run."
        )

    video_source = int(args.input) if args.input.isdigit() else args.input

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open input '{args.input}'. If this is a webcam, check "
            "VirtualBox Devices -> Webcams is enabled for this VM session, and "
            "run `ls -l /dev/video*` to confirm the device node exists."
        )

    # MJPEG first, then resolution: VirtualBox webcam passthrough is often
    # bandwidth-limited over the virtual USB channel, and raw YUYV at higher
    # resolutions can stall frame delivery. MJPEG is compressed and much
    # lighter to push through, and some drivers only honour a resolution
    # change once the pixel format supports it.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live_camera = isinstance(video_source, int)

    # cap.set() can fail silently if the driver rejects the request — check
    # what was actually negotiated instead of assuming it worked.
    if (frame_w, frame_h) != (args.width, args.height):
        print(f"WARNING: requested {args.width}x{args.height} but camera is "
              f"actually delivering {frame_w}x{frame_h}. The driver ignored "
              "the request — run `v4l2-ctl --device=/dev/video0 "
              "--list-formats-ext` to see what it actually supports.")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))

    landmarker = build_landmarker(
        args.model, args.num_people,
        args.min_detection_confidence, args.min_presence_confidence, args.min_tracking_confidence,
    )

    # Mutable click state, written by the mouse callback and read/cleared in
    # the main loop each frame.
    click_state = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["xy"] = (x, y)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # Leader tracking state. "locked" means a leader has been selected and
    # we're actively trying to keep following the same person; "centroid"
    # is that person's last known bbox center, used to match them against
    # this frame's detections (MediaPipe doesn't give a persistent person
    # ID across frames on its own — this is what makes selection "stick").
    leader_state = {"locked": False, "centroid": None, "lost_frames": 0}

    all_frame_landmarks = []
    frame_index = 0
    start_time = time.time()
    last_fps_check = start_time
    fps_frame_count = 0
    display_fps = 0.0

    print(f"Running on '{args.input}' at {frame_w}x{frame_h} (requested; actual capture size can "
          "differ depending on what the camera driver honours). Detecting up to "
          f"{args.num_people} people.")
    print("Click a person in the preview window to select them as the leader. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Live webcam: timestamps must come from the wall clock since
            # frame arrival isn't tied to a fixed fps. Video file: derive
            # from fps so playback timing stays consistent.
            if is_live_camera:
                timestamp_ms = int((time.time() - start_time) * 1000)
            else:
                timestamp_ms = int((frame_index / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Build a flat list of this frame's detections up front, since
            # both click-handling and leader-matching need to scan them.
            detections = []
            for landmarks in result.pose_landmarks:
                bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)
                detections.append({
                    "landmarks": landmarks,
                    "bbox": bbox,
                    "centroid": bbox_centroid(bbox),
                })

            # Handle a pending click: select whichever detected person
            # contains the click point as the new leader. A click always
            # overrides the current leader, so you can retarget on the fly.
            if click_state["xy"] is not None:
                cx, cy = click_state["xy"]
                click_state["xy"] = None
                best_area, best_detection = None, None
                for d in detections:
                    x_min, y_min, x_max, y_max = d["bbox"]
                    if x_min <= cx <= x_max and y_min <= cy <= y_max:
                        area = bbox_area(d["bbox"])
                        # If the click lands in more than one overlapping
                        # box, prefer the smallest (most specific) one.
                        if best_area is None or area < best_area:
                            best_area, best_detection = area, d
                if best_detection is not None:
                    leader_state["locked"] = True
                    leader_state["centroid"] = best_detection["centroid"]
                    leader_state["lost_frames"] = 0
                # A click that misses everyone is a no-op — current
                # leader (if any) keeps being tracked.

            # Match the tracked leader against this frame's detections by
            # nearest centroid. This is deliberately simple (no Kalman
            # filter, no appearance features) — per the team's own build
            # order, escalate only if testing shows plain centroid matching
            # isn't robust enough.
            leader_detection = None
            if leader_state["locked"] and detections:
                best_dist, best_detection = None, None
                for d in detections:
                    dx = d["centroid"][0] - leader_state["centroid"][0]
                    dy = d["centroid"][1] - leader_state["centroid"][1]
                    dist = math.hypot(dx, dy)
                    if best_dist is None or dist < best_dist:
                        best_dist, best_detection = dist, d
                if best_dist is not None and best_dist <= args.max_match_distance:
                    leader_detection = best_detection
                    leader_state["centroid"] = leader_detection["centroid"]
                    leader_state["lost_frames"] = 0
                else:
                    leader_state["lost_frames"] += 1
                    if leader_state["lost_frames"] > args.leader_lost_frames:
                        leader_state["locked"] = False
                        leader_state["centroid"] = None
                        leader_state["lost_frames"] = 0

            # Draw. The leader gets a full gold box, label, and skeleton.
            # Everyone else is drawn dimmed with no skeleton and no
            # label — visible enough to show multi-person detection is
            # working, but visually "ignored" and excluded from export.
            for d in detections:
                x_min, y_min, x_max, y_max = d["bbox"]
                if leader_detection is not None and d is leader_detection:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 215, 255), 3)
                    cv2.putText(frame, "Leader", (x_min, max(y_min - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
                    draw_skeleton(frame, d["landmarks"], frame_w, frame_h, color=(0, 200, 0))
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (90, 90, 90), 1)

            if leader_state["locked"] and leader_detection is None:
                cv2.putText(frame, "Leader lost - hold on...", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif not leader_state["locked"]:
                cv2.putText(frame, "Click a person to select leader", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Export is leader-only, on purpose: everyone else is "ignored"
            # for the data hand-off, not just visually dimmed.
            if args.export_landmarks and leader_detection is not None:
                x_min, y_min, x_max, y_max = leader_detection["bbox"]
                all_frame_landmarks.append({
                    "frame_id": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "leader_id": 0,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "landmarks": [
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                        for lm in leader_detection["landmarks"]
                    ],
                })

            # Rolling FPS counter, refreshed once a second — lets you see
            # live whether a resolution/model change actually fixed the lag,
            # instead of eyeballing it.
            fps_frame_count += 1
            now = time.time()
            if now - last_fps_check >= 1.0:
                display_fps = fps_frame_count / (now - last_fps_check)
                fps_frame_count = 0
                last_fps_check = now
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1
    except KeyboardInterrupt:
        # Caught here (not left to propagate) so cleanup still runs AND the
        # summary below still prints — Ctrl-C during a benchmark run should
        # give you real numbers, not just a bare traceback.
        print("\nInterrupted by user (Ctrl-C).")
    finally:
        # Always release resources, even on Ctrl-C or an unexpected error —
        # otherwise the camera device can stay locked for the next run.
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        landmarker.close()

    elapsed = time.time() - start_time
    avg_fps = frame_index / elapsed if elapsed > 0 else 0.0
    print(f"Done. Processed {frame_index} frames in {elapsed:.1f}s ({avg_fps:.2f} fps average).")
    if args.export_landmarks:
        with open(args.export_landmarks, "w") as f:
            json.dump(all_frame_landmarks, f, indent=2)
        print(f"Leader-only landmarks exported to {args.export_landmarks}")


if __name__ == "__main__":
    main()
