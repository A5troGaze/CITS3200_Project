"""
Multi-Person Pose Detection with Leader Selection and 3D World-Landmark Export

Features:
  - Detects multiple people in a webcam stream or video file.
  - Click a detected person to select them as the leader.
  - Tracks the selected leader using nearest bounding-box centroid matching.
  - Draws the leader prominently and other detections as dimmed boxes.
  - Exports only the selected leader's 33 MediaPipe pose_world_landmarks.
  - Allows a configurable grace period when the leader is temporarily lost.

The exported world landmarks are MediaPipe's estimated 3D positions in metres,
relative to the midpoint of the hips. They are intended as input for downstream
pose retargeting and inverse kinematics; they are not robot joint angles.

Examples:
  python3 pose_test-multi-world-landmarks.py --model /path/to/pose_landmarker.task

  python3 pose_test-multi-world-landmarks.py \
      --input test_clip.mp4 \
      --model /path/to/pose_landmarker.task \
      --output annotated.mp4 \
      --export-landmarks leader_world_landmarks.json
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
WINDOW_NAME = "Multi-Person Pose Tracking - click leader, 'q' to quit"


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def build_landmarker(
    model_path,
    num_people,
    min_detection_confidence,
    min_presence_confidence,
    min_tracking_confidence,
):
    """Create a multi-person MediaPipe Pose Landmarker in video mode."""
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
    """Convert normalized image landmarks to a clamped pixel bounding box."""
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
    """Draw connected skeleton lines and joint points."""
    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, color, -1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-person detection with click-to-select leader tracking and "
            "leader-only 3D world-landmark export"
        )
    )
    parser.add_argument(
        "--input",
        default="0",
        help="Camera index (default 0) or path to a video file",
    )
    parser.add_argument("--output", default=None, help="Optional annotated video path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe .task pose-landmarker model",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height")
    parser.add_argument(
        "--export-landmarks",
        default=None,
        help="Optional JSON path for leader-only world landmarks",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Unsupported because leader selection requires clicking the preview",
    )
    parser.add_argument(
        "--num-people",
        type=int,
        default=4,
        help="Maximum number of people to detect per frame",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum pose-detection confidence",
    )
    parser.add_argument(
        "--min-presence-confidence",
        type=float,
        default=0.5,
        help="Minimum pose-presence confidence",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe tracking confidence",
    )
    parser.add_argument(
        "--max-match-distance",
        type=float,
        default=120.0,
        help="Maximum centroid distance in pixels for matching the selected leader",
    )
    parser.add_argument(
        "--leader-lost-frames",
        type=int,
        default=15,
        help="Frames the leader may be unmatched before selection is cleared",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_preview:
        raise SystemExit(
            "Leader selection requires the preview window. Remove --no-preview to run."
        )
    if args.num_people < 1:
        raise SystemExit("--num-people must be at least 1")
    if args.max_match_distance < 0:
        raise SystemExit("--max-match-distance cannot be negative")
    if args.leader_lost_frames < 0:
        raise SystemExit("--leader-lost-frames cannot be negative")

    video_source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open input '{args.input}'.")

    # MJPEG helps reduce bandwidth pressure for some VM webcam configurations.
    if isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live_camera = isinstance(video_source, int)

    if (frame_w, frame_h) != (args.width, args.height):
        print(
            f"WARNING: requested {args.width}x{args.height}, but the input is "
            f"delivering {frame_w}x{frame_h}."
        )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            cap.release()
            raise SystemExit(f"Could not open output video '{args.output}'.")

    landmarker = build_landmarker(
        args.model,
        args.num_people,
        args.min_detection_confidence,
        args.min_presence_confidence,
        args.min_tracking_confidence,
    )

    click_state = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state["xy"] = (x, y)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # MediaPipe does not provide persistent person IDs. This prototype keeps
    # the selected leader by matching the closest bounding-box centroid.
    leader_state = {"locked": False, "centroid": None, "lost_frames": 0}

    all_frame_landmarks = []
    frame_index = 0
    start_time = time.time()
    last_fps_check = start_time
    fps_frame_count = 0
    display_fps = 0.0

    print(
        f"Running on '{args.input}' at {frame_w}x{frame_h}; detecting up to "
        f"{args.num_people} people."
    )
    print("Click a person to select the leader. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            if is_live_camera:
                timestamp_ms = int((time.time() - start_time) * 1000)
            else:
                timestamp_ms = int((frame_index / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            detections = []
            world_results = result.pose_world_landmarks
            for pose_index, landmarks in enumerate(result.pose_landmarks):
                # Pose and world-landmark lists use the same pose ordering.
                world_landmarks = (
                    world_results[pose_index]
                    if pose_index < len(world_results)
                    else None
                )
                bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)
                detections.append(
                    {
                        "landmarks": landmarks,
                        "world_landmarks": world_landmarks,
                        "bbox": bbox,
                        "centroid": bbox_centroid(bbox),
                    }
                )

            # A click selects the smallest detected bounding box containing it.
            if click_state["xy"] is not None:
                click_x, click_y = click_state["xy"]
                click_state["xy"] = None
                candidates = []
                for detection in detections:
                    x_min, y_min, x_max, y_max = detection["bbox"]
                    if x_min <= click_x <= x_max and y_min <= click_y <= y_max:
                        candidates.append(detection)
                if candidates:
                    selected = min(candidates, key=lambda d: bbox_area(d["bbox"]))
                    leader_state["locked"] = True
                    leader_state["centroid"] = selected["centroid"]
                    leader_state["lost_frames"] = 0

            leader_detection = None
            if leader_state["locked"]:
                if detections:
                    nearest = min(
                        detections,
                        key=lambda d: math.hypot(
                            d["centroid"][0] - leader_state["centroid"][0],
                            d["centroid"][1] - leader_state["centroid"][1],
                        ),
                    )
                    nearest_distance = math.hypot(
                        nearest["centroid"][0] - leader_state["centroid"][0],
                        nearest["centroid"][1] - leader_state["centroid"][1],
                    )
                    if nearest_distance <= args.max_match_distance:
                        leader_detection = nearest
                        leader_state["centroid"] = nearest["centroid"]
                        leader_state["lost_frames"] = 0

                # Count every unmatched frame, including frames with no poses.
                if leader_detection is None:
                    leader_state["lost_frames"] += 1
                    if leader_state["lost_frames"] > args.leader_lost_frames:
                        leader_state["locked"] = False
                        leader_state["centroid"] = None
                        leader_state["lost_frames"] = 0

            for detection in detections:
                x_min, y_min, x_max, y_max = detection["bbox"]
                if leader_detection is not None and detection is leader_detection:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 215, 255), 3)
                    cv2.putText(
                        frame,
                        "Leader",
                        (x_min, max(y_min - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 215, 255),
                        2,
                    )
                    draw_skeleton(
                        frame,
                        detection["landmarks"],
                        frame_w,
                        frame_h,
                        color=(0, 200, 0),
                    )
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (90, 90, 90), 1)

            if leader_state["locked"] and leader_detection is None:
                cv2.putText(
                    frame,
                    "Leader temporarily lost...",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            elif not leader_state["locked"]:
                cv2.putText(
                    frame,
                    "Click a person to select leader",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            if args.export_landmarks and leader_detection is not None:
                world_landmarks = leader_detection["world_landmarks"]
                if world_landmarks is not None:
                    x_min, y_min, x_max, y_max = leader_detection["bbox"]
                    all_frame_landmarks.append(
                        {
                            "frame_id": frame_index,
                            "timestamp_ms": timestamp_ms,
                            "leader_id": 0,
                            "bbox": [x_min, y_min, x_max, y_max],
                            "landmarks_world_m": [
                                {
                                    "x": lm.x,
                                    "y": lm.y,
                                    "z": lm.z,
                                    "visibility": lm.visibility,
                                }
                                for lm in world_landmarks
                            ],
                        }
                    )

            fps_frame_count += 1
            now = time.time()
            if now - last_fps_check >= 1.0:
                display_fps = fps_frame_count / (now - last_fps_check)
                fps_frame_count = 0
                last_fps_check = now
            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if writer is not None:
                writer.write(frame)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl-C).")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        landmarker.close()

    elapsed = time.time() - start_time
    avg_fps = frame_index / elapsed if elapsed > 0 else 0.0
    print(
        f"Done. Processed {frame_index} frames in {elapsed:.1f}s "
        f"({avg_fps:.2f} fps average)."
    )
    if args.export_landmarks:
        with open(args.export_landmarks, "w") as output_file:
            json.dump(all_frame_landmarks, output_file, indent=2)
        print(f"Leader-only world landmarks exported to {args.export_landmarks}")


if __name__ == "__main__":
    main()
