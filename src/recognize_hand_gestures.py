'''
recognize_hand_gestures.py - test tool for checking hand gesture recognition.

Reads the webcam (or a video file), detects the hand and prints the recognised
gesture name to the console whenever it changes. Use it after recording
gestures with record_hand_gestures.py to check they are recognised correctly,
without starting the simulator or the robot.

Usage (run from the 'Project/' folder):
    python src/recognize_hand_gestures.py                                    # live camera
    python src/recognize_hand_gestures.py --video clip.mp4                   # pre-recorded video file
    python src/recognize_hand_gestures.py --video clip.mp4 --threshold 3.0   # looser matching (videos often need this)

Keys (in the camera window, if a display is available):
    q   quit (or press Ctrl+C in the terminal)
'''

#----------------------------------------------------------
# Imports (standard library + third party)
#----------------------------------------------------------
import argparse
import cv2
import json
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_recognition import landmarks_to_vector, classify, draw_skeleton


#----------------------------------------------------------
# Paths
#----------------------------------------------------------
# All paths are built from this file's location, so the project works wherever it is installed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "Dependencies"))   # Sits next to the project folder

HAND_MODEL_PATH = os.path.join(DEPENDENCIES_DIR, "Models", "hand_landmarker.task")  # MediaPipe hand model (not in the repo)
HAND_MODEL_DATA_PATH = os.path.join(BASE_DIR, "data", "gestures.json")              # Recorded gesture samples


#----------------------------------------------------------
# Command line arguments
#----------------------------------------------------------
# Default is the live camera (index 0). Pass --video <path> to test against a
# pre-recorded file instead (useful if your machine/VM has no camera access).
# The rest of the script behaves identically either way.
parser = argparse.ArgumentParser()
parser.add_argument("--video", default=None, help="Path to a video file to use instead of the live camera")
parser.add_argument("--threshold", type=float, default=1.5, help="Gesture matching threshold (default 1.5). Higher = looser matching, lower = tighter.")
args = parser.parse_args()


#----------------------------------------------------------
# Load the recorded gestures
#----------------------------------------------------------
if not os.path.exists(HAND_MODEL_DATA_PATH):
    raise FileNotFoundError(
        f"Couldn't find {HAND_MODEL_DATA_PATH}. Make sure data/gestures.json (recorded with "
        "record_hand_gestures.py) is present before running recognition."
    )

with open(HAND_MODEL_DATA_PATH) as f:
    registry = json.load(f)


#----------------------------------------------------------
# Set up the hand landmarker model
#----------------------------------------------------------
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)   # Set path to model
options = vision.HandLandmarkerOptions(                               # Configure model:
    base_options=base_options,                                            # Path
    num_hands=1,                                                          # Number of hands
    running_mode=vision.RunningMode.VIDEO                                 # Expect video stream
)
landmarker = vision.HandLandmarker.create_from_options(options)       # Create the usable object


#----------------------------------------------------------
# Camera / video setup
#----------------------------------------------------------
cap = cv2.VideoCapture(args.video if args.video else 0)    # Video file if given, otherwise the default webcam
frame_timestamp_ms = 0                                     # MediaPipe video mode needs increasing timestamps (~30 fps)

# The display is optional: some machines (e.g. headless VMs) can't open a GUI window.
# On Linux, no DISPLAY variable means there is no X11/GUI at all. Calling cv2.imshow()
# in that case crashes the whole process (it can't be caught as an exception), so we
# check for this BEFORE trying to open a window, rather than after.
display_enabled = True
if os.name == "posix" and "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    display_enabled = False
    print("(No display detected — running in console-only mode.)")

last_gesture = None                 # Previous frame's result, so we only print when it changes
best_overall_dist = float("inf")    # Smallest distance seen in the whole run, even on frames that never beat the threshold

print("Recognizing gestures from " + ("camera (index 0)" if not args.video else f"video file: {args.video}"))
print(f"Gesture threshold: {args.threshold}")
print("Press 'q' to quit (if a display window is open), or Ctrl+C in the terminal.")


#----------------------------------------------------------
# Main loop
#----------------------------------------------------------
while cap.isOpened():
    # Grab a frame, mirror it (as in the recorder, so gestures match) and detect the hand
    ok, frame = cap.read()
    if not ok:                                              # End of video file, or camera lost
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # MediaPipe expects RGB, OpenCV gives BGR
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    draw_skeleton(frame, result)

    #------------------------------------------------------
    # Classify whatever hand was found this frame
    #------------------------------------------------------
    if result.hand_landmarks:
        vector = landmarks_to_vector(result.hand_landmarks[0])                  # Hand landmarks -> feature vector
        name, dist = classify(vector, registry, threshold=args.threshold)       # Closest recorded gesture (name is None if too far)
        current_gesture = name if name else "no match"
        best_overall_dist = min(best_overall_dist, dist)                        # Track how close this frame got, matched or not
    else:
        current_gesture = "no hand"
        dist = None

    #------------------------------------------------------
    # TEMPORARY DEBUG: print the distance a few times a second, whether or not the gesture changed
    #------------------------------------------------------
    if dist is not None and frame_timestamp_ms % 300 < 33:
        print(f"  live: {current_gesture} dist={dist:.3f}")

    #------------------------------------------------------
    # Only print when the recognised gesture actually changes
    #------------------------------------------------------
    if current_gesture != last_gesture:
        if dist is not None:
            print(f"[{frame_timestamp_ms}ms] {current_gesture} (distance={dist:.3f})")
        else:
            print(f"[{frame_timestamp_ms}ms] {current_gesture}")
        last_gesture = current_gesture

    #------------------------------------------------------
    # Optional display: label the frame and show it, if a display is available
    #------------------------------------------------------
    if display_enabled:
        label = current_gesture if current_gesture not in ("no hand",) else "..."
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        try:
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # No GUI available (e.g. headless VM) — keep running console-only
            print("(No display available — continuing in console-only mode.)")
            display_enabled = False


#----------------------------------------------------------
# Summary: how close did we get, even if nothing matched?
#----------------------------------------------------------
if best_overall_dist < float("inf"):
    print(f"Closest distance seen across the whole run: {best_overall_dist:.3f}")
else:
    print("No hand was ever detected in this run.")


#----------------------------------------------------------
# Cleanup
#----------------------------------------------------------
cap.release()
landmarker.close()
if display_enabled:
    cv2.destroyAllWindows()