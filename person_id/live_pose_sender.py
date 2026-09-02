"""
Windows live webcam pose sender.

Run this on Windows, not WSL, when WSL cannot access the webcam.

It reads webcam frames, detects one person's MediaPipe world landmarks
using the MediaPipe Tasks API, and sends the 3D landmark data over UDP
to the WSL receiver.
"""

import argparse
import json
import socket
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def landmark_to_dict(lm):
    return {
        "x": lm.x,
        "y": lm.y,
        "z": lm.z,
        "visibility": getattr(lm, "visibility", 0.0),
    }


def build_landmarker(model_path):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)

    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    return mp_vision.PoseLandmarker.create_from_options(options)


def draw_landmarks(frame, pose_landmarks):
    height, width = frame.shape[:2]

    for lm in pose_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def main():
    script_dir = Path(__file__).resolve().parent
    default_model = script_dir / "pose_landmarker_full.task"

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", required=True, help="WSL IP address")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--model", default=str(default_model))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Pose model not found: {args.model}\n"
            "Pass the model path with --model if it is somewhere else."
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    landmarker = build_landmarker(args.model)

    frame_id = 0
    print(f"Sending live pose landmarks to {args.host}:{args.port}")
    print(f"Using model: {args.model}")
    print("Press q in the camera window to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break

            frame_id += 1
            timestamp_ms = int(time.time() * 1000)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            if result.pose_world_landmarks:
                world_landmarks = result.pose_world_landmarks[0]

                packet = {
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "landmarks_world_m": [
                        landmark_to_dict(lm)
                        for lm in world_landmarks
                    ],
                }

                data = json.dumps(packet).encode("utf-8")
                sock.sendto(data, (args.host, args.port))

                if frame_id % 30 == 0:
                    print(
                        f"Sent frame {frame_id}; "
                        f"landmarks={len(world_landmarks)}"
                    )

            if result.pose_landmarks:
                draw_landmarks(frame, result.pose_landmarks[0])

            cv2.imshow("Live Pose Sender", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        sock.close()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
