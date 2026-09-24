import sys
import json
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_core import landmarks_to_vector, classify, draw_skeleton
from simulation_controller import SimController
from real_controller import RealController
from webcam_source import WebcamSource
# RealsenseSource is imported lazily inside build_camera() -- pyrealsense2
# has no official macOS wheels, so importing it here would break running
# in "sim" mode on a machine (like a laptop) that will never touch the
# real robot's camera.



#== Define Paths =====================================================================
HAND_MODEL_PATH = os.path.expanduser("~/CITS3200/Dependencies/Models/hand_landmarker.task")
HAND_MODEL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "gestures.json")

for i in [HAND_MODEL_DATA_PATH, HAND_MODEL_PATH]:                   # For each path
    if not os.path.exists(i):                                       # If path does not exist; Raise Error
        raise FileNotFoundError(
            f"Couldn't find {i}. Make sure {i} is present before running again."
        )

#==
def build_controller(backend):
    if backend == "sim":
        return SimController()
    if backend == "real":
        return RealController()
    raise ValueError(f"Unknown backend: {backend}.\nOptions: 'sim' or 'real'.")


def build_camera(backend):
    if backend == "sim":
        return WebcamSource()
    if backend == "real":
        # Imported here, not at the top of the file, so that running in
        # "sim" mode never requires pyrealsense2 to be installed (it has
        # no official macOS wheel -- see realsense_source.py).
        from realsense_source import RealsenseSource
        return RealsenseSource()
    raise ValueError(f"Unknown backend: {backend}.\nOptions: 'sim' or 'real'.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("sim", "real"):
        print("Usage: python gesture_control.py [sim|real]")
        sys.exit(1)


    controller = build_controller(sys.argv[1])
    controller.init()
    controller.start()

    camera = build_camera(sys.argv[1])
    camera.init()

    with open(HAND_MODEL_DATA_PATH) as f:
        registry = json.load(f)

    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(                             # Define Model:
        base_options = base_options,                                    # Path
        num_hands = 1,                                                  # Number of hands for gestures
        running_mode = vision.RunningMode.VIDEO                         # Video Stream
    )
    gesture_model = vision.HandLandmarker.create_from_options(options)  # Create useable object from definition

    frame_timestamp_ms = 0

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += 33
        result = gesture_model.detect_for_video(mp_image, frame_timestamp_ms)

        gesture_name = None
        if result.hand_landmarks:
            vector = landmarks_to_vector(result.hand_landmarks[0])
            gesture_name, dist = classify(vector, registry)

        controller.set_gesture(gesture_name)

        draw_skeleton(frame, result)
        cv2.putText(frame, f"Gesture: {gesture_name or 'no match'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    controller.stop()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()