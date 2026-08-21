"""
Single-Person Identification — finalized
CITS3200 Humanoid Project | Person Identification subgroup

What this does:
  - Reads a live camera feed (or a video file)
  - Detects ONE person (the "leader") per frame using MediaPipe Pose Landmarker
  - Draws a bounding box + full skeleton (connected lines, not just dots) around them
  - Shows a live FPS counter so lag is visible while testing, not guessed at
  - Optionally writes the annotated stream to a video file
  - Optionally exports per-frame landmark data as JSON, using the hand-off
    schema the kinematics/gesture team will need: {frame_id, bbox, landmarks[33], timestamp}

Before running:
  1. source ~/CITS3200/Dependencies/g1-env/bin/activate
  2. Model file expected at DEFAULT_MODEL_PATH below (already downloaded per team setup).
     If yours lives elsewhere, pass --model /path/to/pose_landmarker.task

Usage:
  # Live webcam, on-screen preview only:
  python3 pose_test.py

  # Live webcam, also write annotated output + landmarks:
  python3 pose_test.py --output out.mp4 --export-landmarks out.json

  # From a video file instead of webcam:
  python3 pose_test.py --input test_clip.mp4 --output out.mp4

  # If lag is still an issue, drop resolution further:
  python3 pose_test.py --width 480 --height 360
"""

import argparse
import json
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL_PATH = "/home/ubuntu/CITS3200/models/pose_landmarker.task"

# MediaPipe's documented skeletal connections for the 33 pose landmarks —
# drawing these (not just dots) is what makes the overlay actually read as
# a skeleton in the demo, matching what the client asked for.
# Hardcoded here (rather than pulled from mp.solutions.pose) because
# Tasks-API-only MediaPipe installs don't always expose the legacy
# `solutions` submodule.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def build_landmarker(model_path):
    """Single-person MediaPipe PoseLandmarker in VIDEO mode."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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


def draw_skeleton(frame, landmarks, frame_w, frame_h):
    """Draw connected skeleton lines plus joint dots."""
    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (0, 200, 0), 2)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser(description="Single-person identification (bounding box + skeleton)")
    parser.add_argument("--input", default="0", help="Camera index (default 0) or path to a video file")
    parser.add_argument("--output", default=None, help="Optional path to write annotated video")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the .task pose landmarker model")
    parser.add_argument("--width", type=int, default=640, help="Capture width (lower = faster, less laggy)")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--export-landmarks", default=None, help="Optional path to dump per-frame landmarks as JSON")
    parser.add_argument("--no-preview", action="store_true", help="Don't open a live preview window (headless)")
    args = parser.parse_args()

    video_source = int(args.input) if args.input.isdigit() else args.input

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open input '{args.input}'. If this is a webcam, check "
            "VirtualBox Devices -> Webcams is enabled for this VM session, and "
            "run `ls -l /dev/video*` to confirm the device node exists."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live_camera = isinstance(video_source, int)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))

    landmarker = build_landmarker(args.model)

    all_frame_landmarks = []
    frame_index = 0
    start_time = time.time()
    last_fps_check = start_time
    fps_frame_count = 0
    display_fps = 0.0

    print(f"Running on '{args.input}' at {frame_w}x{frame_h} (requested; actual capture size can "
          "differ depending on what the camera driver honours).")
    if not args.no_preview:
        print("Press 'q' in the preview window to quit.")

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

            if result.pose_landmarks:
                # pose_landmarks = image-relative coordinates (0.0-1.0, plus a
                # rough relative depth). Good for drawing on screen, since it
                # maps directly to pixel positions.
                landmarks = result.pose_landmarks[0]  # single-person: only ever one entry
                x_min, y_min, x_max, y_max = landmarks_to_bbox(landmarks, frame_w, frame_h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 215, 255), 2)
                cv2.putText(frame, "Leader", (x_min, max(y_min - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
                draw_skeleton(frame, landmarks, frame_w, frame_h)

                if args.export_landmarks:
                    # pose_world_landmarks = REAL 3D coordinates in metres,
                    # anchored to the hip midpoint, independent of how close
                    # the person is to the camera. This is what the client
                    # asked for in the Week 5 meeting: actual joint dimensions
                    # for mapping onto the robot's degrees of freedom, not
                    # just where the joint appears on screen. This is what
                    # gets exported — the on-screen drawing above still uses
                    # the image-relative landmarks, since that's what pixel
                    # positions need.
                    world_landmarks = result.pose_world_landmarks[0]

                    all_frame_landmarks.append({
                        "frame_id": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "leader_id": 0,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "landmarks_world_m": [
                            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                            for lm in world_landmarks
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

            if not args.no_preview:
                cv2.imshow("Single-Person Identification (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        # Always release resources, even on Ctrl-C or an unexpected error —
        # otherwise the camera device can stay locked for the next run.
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        landmarker.close()

    print(f"Done. Processed {frame_index} frames.")
    if args.export_landmarks:
        with open(args.export_landmarks, "w") as f:
            json.dump(all_frame_landmarks, f, indent=2)
        print(f"Landmarks exported to {args.export_landmarks}")


if __name__ == "__main__":
    main()