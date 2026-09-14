"""Select and track a leader, then drive their arm pose in G1 MuJoCo.

MediaPipe detects up to ``--num-people`` poses. The tester clicks a detected
person to select the leader; subsequent frames match that leader using bounding-
box centroid distance. Only the selected leader's 33 world landmarks are
retargeted. Use ``--dry-run`` to test detection without CycloneDDS or MuJoCo.
"""

import argparse
import math
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from mujoco_pose_controller import MujocoPoseController
from pose_retargeting import mediapipe_landmarks_to_xyz, retarget_arms_indexed


WINDOW_NAME = "Leader Pose Control - click leader, q to quit"
DEFAULT_MODEL = os.path.expanduser("~/CITS3200/Dependencies/Models/pose_landmarker.task")

def landmarks_to_bbox(landmarks, width, height, padding=20):
    # Convert normalized image landmarks to a pixel box clamped to frame edges.
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    return (
        max(int(min(xs)) - padding, 0),
        max(int(min(ys)) - padding, 0),
        min(int(max(xs)) + padding, width),
        min(int(max(ys)) + padding, height),
    )


def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_args():
    parser = argparse.ArgumentParser(description="Leader pose to G1 MuJoCo control")
    parser.add_argument("--input", default="0", help="Camera index or video path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-people", type=int, default=4)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--max-match-distance", type=float, default=120.0)
    parser.add_argument("--lost-frames", type=int, default=15)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection and angle estimation without CycloneDDS/MuJoCo",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.input) if args.input.isdigit() else args.input
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open input: {args.input}")

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=args.num_people,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    controller = None
    if not args.dry_run:
        # Dry-run mode skips DDS entirely and exercises only detection and maths.
        controller = MujocoPoseController(args.domain_id, args.interface)
        print("Waiting for MuJoCo low-state messages...")
        try:
            controller.init()
        except TimeoutError as exc:
            landmarker.close()
            capture.release()
            raise SystemExit(f"Could not connect to MuJoCo: {exc}")
        controller.start()

    click = {"point": None}
    leader = {"centroid": None, "lost": 0}

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            click["point"] = (x, y)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    started = time.monotonic()
    was_tracking = False  # Tracks whether a command must be cleared on pose loss.

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Camera timestamps use elapsed time; files use their frame rate.
            timestamp = (
                int((time.monotonic() - started) * 1000)
                if isinstance(source, int)
                else int(frame_index / fps * 1000)
            )
            result = landmarker.detect_for_video(image, timestamp)

            detections = []
            for index, image_landmarks in enumerate(result.pose_landmarks):
                world = result.pose_world_landmarks[index] if index < len(result.pose_world_landmarks) else None
                box = landmarks_to_bbox(image_landmarks, width, height)
                detections.append({"bbox": box, "centroid": centroid(box), "world": world})

            if click["point"] is not None:
                # A new click deliberately replaces any currently selected leader.
                x, y = click["point"]
                click["point"] = None
                choices = [d for d in detections if d["bbox"][0] <= x <= d["bbox"][2]
                           and d["bbox"][1] <= y <= d["bbox"][3]]
                if choices:
                    chosen = min(choices, key=lambda d: distance(d["centroid"], (x, y)))
                    leader = {"centroid": chosen["centroid"], "lost": 0}

            selected = None
            if leader["centroid"] is not None and detections:
                # Re-identify the leader using the nearest detected box centroid.
                candidate = min(detections, key=lambda d: distance(d["centroid"], leader["centroid"]))
                if distance(candidate["centroid"], leader["centroid"]) <= args.max_match_distance:
                    selected = candidate
                    leader = {"centroid": candidate["centroid"], "lost": 0}

            if leader["centroid"] is not None and selected is None:
                # Preserve identity briefly for reacquisition after an occlusion.
                # Arm targets are still cleared immediately while pose data is absent.
                leader["lost"] += 1
                if leader["lost"] > args.lost_frames:
                    leader = {"centroid": None, "lost": 0}
                    if controller:
                        controller.clear_targets()
                    was_tracking = False

            for detection in detections:
                colour = (0, 215, 255) if detection is selected else (100, 100, 100)
                cv2.rectangle(frame, detection["bbox"][:2], detection["bbox"][2:], colour, 2)

            if selected and selected["world"]:
                try:
                    xyz = mediapipe_landmarks_to_xyz(selected["world"])
                    targets = retarget_arms_indexed(xyz)
                except ValueError as exc:
                    # Never publish malformed or non-finite landmark-derived targets.
                    print(f"Frame {frame_index} skipped: {exc}")
                    if controller and was_tracking:
                        controller.clear_targets()
                    was_tracking = False
                else:
                    if controller:
                        controller.set_targets(targets)
                    was_tracking = True
                    summary = " ".join(f"{index}:{angle:+.2f}" for index, angle in sorted(targets.items()))
                    cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 255, 255), 1)
            elif controller and was_tracking:
                # No valid selected pose: return the controlled joints toward startup.
                controller.clear_targets()
                was_tracking = False

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    finally:
        # Ensure simulator commands stop and the arms return even after Ctrl-C/error.
        if controller:
            controller.return_to_start()
            controller.stop()
        landmarker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
