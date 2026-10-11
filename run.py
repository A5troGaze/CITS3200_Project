'''
run.py - entry point for the Humanoid Control system.

Reads the webcam, recognises hand gestures (gesture mode) or copies the
user's arm movements (mimic mode) and sends the result to the G1, either in
the simulator or on the real robot.

Usage:
    python run.py sim              # simulator, default gesture threshold (1.5)
    python run.py sim 2.0          # simulator, custom gesture threshold
    python run.py real             # real robot

Keys (in the camera window):
    m   switch between gesture mode and mimic mode
    q   quit

This file used to be called: gesture_control.py
'''

#----------------------------------------------------------
# Imports (standard library + third party)
#----------------------------------------------------------
import os
import sys
import json
import cv2
import math
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#----------------------------------------------------------
# Paths
#----------------------------------------------------------
# All paths are built from this file's location, so the project works wherever it is installed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)                 # Project modules live in `RUN/`; Python must search there for imports
DEPENDENCIES_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "Dependencies"))   # Sits next to the project folder

HAND_MODEL_PATH = os.path.join(DEPENDENCIES_DIR, "Models", "hand_landmarker.task")  # MediaPipe hand model (not in the repo)
HAND_MODEL_DATA_PATH = os.path.join(SRC_DIR, "data", "gestures.json")               # Gesture definitions


#----------------------------------------------------------
# Project imports (must come after the `sys.path` line above)
#----------------------------------------------------------
from gesture_recognition import landmarks_to_vector, classify, draw_skeleton
from command_source import PlaceholderMimicSource


#----------------------------------------------------------
# Constants
#----------------------------------------------------------
MODE_GESTURE = "gesture"                    # Hand gestures drive the robot's walking
MODE_MIMIC = "mimic"                        # The robot's arms copy the user's arms
MODE_TOGGLE_KEY = ord("m")                  # Key that switches between the two modes
BACKENDS = ("sim", "real")                  # Where commands are sent: simulator or real robot
DEFAULT_GESTURE_THRESHOLD = 1.5             # Used when no threshold is given on the command line


#----------------------------------------------------------
# Helper functions
#----------------------------------------------------------
def parse_args():
    '''Read the command line: backend (required) and gesture threshold (optional).'''
    parser = argparse.ArgumentParser(description="Gesture-driven control of the Unitree G1.")
    parser.add_argument("backend", choices=BACKENDS, help="'sim' for the simulator, 'real' for the robot")
    parser.add_argument("threshold", nargs="?", type=float, default=DEFAULT_GESTURE_THRESHOLD,
                        help=f"Gesture matching threshold (default {DEFAULT_GESTURE_THRESHOLD}). Higher = looser matching, lower = tighter.")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold <= 0:    # Reject 0, negative numbers and nan
        parser.error("Threshold must be a positive number")
    return args


def build_controller(backend):
    '''Create the controller for the chosen backend. Imported here so only the one in use is loaded.'''
    if backend == "sim":
        from simulation_controller import SimController
        return SimController()
    if backend == "real":
        from real_controller import RealController
        return RealController()

    raise ValueError(f"UNKNOWN BACKEND: {backend}.\nOptions: {', '.join(BACKENDS)}.")


def build_mimic_source():
    '''Create the mimic source (pose tracking + arm retargeting), or a do-nothing
    placeholder if it can't start, so gesture mode keeps working either way.

    Built on first use rather than at startup because loading the pose model
    and retargeting libraries takes a moment.

    input_flipped=True because the frame passed in has already been mirrored
    with cv2.flip(frame, 1); the source un-flips it before pose detection.'''
    try:
        from mimic_source import MimicSource
        return MimicSource(num_people=1, input_flipped=True)
    except Exception as e:
        print(f"Couldn't start MimicSource: {e}\nMimic mode will do nothing.")
        return PlaceholderMimicSource()


#----------------------------------------------------------
# Main
#----------------------------------------------------------
def main():
    args = parse_args()
    print(f"Backend: {args.backend}, Gesture threshold: {args.threshold}")

    # Check the required files before starting the robot, so a missing file fails early and clearly
    for path in [HAND_MODEL_DATA_PATH, HAND_MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Couldn't find {path}. Check that the Dependencies folder sits next to this project folder "
                f"and contains Models/hand_landmarker.task (see SETUP.md)."
            )

    # Connect to the simulator / robot. start() runs the startup sequence
    # (stand the robot up) and then begins streaming commands.
    controller = build_controller(args.backend)
    controller.init()
    controller.start()

    with open(HAND_MODEL_DATA_PATH) as f:
        registry = json.load(f)             # Known gestures, used by classify() to name what the hand is doing

    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(                             # Define Model:
        base_options = base_options,                                    # Path
        num_hands = 1,                                                  # Number of hands for gestures
        running_mode = vision.RunningMode.VIDEO                         # Video Stream
    )
    gesture_model = vision.HandLandmarker.create_from_options(options)  # Create useable object from definition

    stream = cv2.VideoCapture(0)            # Default webcam
    frame_timestamp_ms = 0                  # Synthetic clock (MediaPipe needs timestamps that always increase)
    mode = MODE_GESTURE
    mimic_source = None                     # built on the first switch to mimic mode

    try:
        while stream.isOpened():
            ok, frame = stream.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)      # Mirror the image so it behaves like a mirror for the user
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # MediaPipe expects RGB; OpenCV gives BGR
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33        # About 30 frames per second

            if mode == MODE_GESTURE:
                # Find the hand, then match its shape against the known gestures
                result = gesture_model.detect_for_video(mp_image, frame_timestamp_ms)
                gesture_name = None

                if result.hand_landmarks:
                    vector = landmarks_to_vector(result.hand_landmarks[0])
                    gesture_name, _ = classify(vector, registry, threshold=args.threshold)

                controller.set_gesture(gesture_name)    # The controller turns the gesture into walking commands
                draw_skeleton(frame, result)
                status_text = f"Gesture: {gesture_name or 'no match'}"

            else:   # MODE_MIMIC
                # Track the user's body and send the matching arm angles to the robot
                output = mimic_source.step(frame, frame_timestamp_ms / 1000.0)
                controller.set_joint_targets(output.joint_targets)
                status_text = f"Mimic mode ({len(output.joint_targets)} joint targets)"

            # On-screen text: current status and the key help
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Mode: {mode} ['m' to toggle, 'q' to quit]", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Humanoid Control", frame)

            key = cv2.waitKey(1) & 0xFF     # Check the keyboard (also lets the window refresh)
            if key == ord('q'):
                break

            if key == MODE_TOGGLE_KEY:
                previous = mode
                mode = MODE_MIMIC if mode == MODE_GESTURE else MODE_GESTURE
                print(f"Switching to {mode} mode")
                try:
                    if mode == MODE_MIMIC:
                        controller.set_gesture(None)   # stop walking before handing over
                        if mimic_source is None:
                            mimic_source = build_mimic_source()     # First time only: slow, the camera freezes briefly
                        mimic_source.reset()                        # Forget the previous session's tracking
                    controller.on_mode_change(mode)                 # Hand the arms over to / back from the mimic code
                except Exception as e:
                    print(f"Mode switch failed ({e}). Staying in {previous} mode")
                    mode = previous
    finally:
        # Always clean up, even after an error, so the robot isn't left mid-command
        close = getattr(mimic_source, "close", None)
        if close is not None:
            close()
        controller.stop()
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()