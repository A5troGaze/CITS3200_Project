'''
record_hand_gestures.py - tool for recording hand gesture samples.

Shows the webcam with the hand skeleton drawn on it. Press a number key (1-8)
while holding a gesture and the script captures 20 samples of it. Press 's' to
save everything to data/gestures.json, which gesture mode in run.py reads.

Usage:
    Run from the 'Project/' folder:
        python src/record_hand_gestures.py

    New samples are ADDED to src/data/gestures.json if it already exists.
    To start from scratch, delete that file first, then record all 8 gestures
    before pressing 's' (a gesture left unrecorded is saved with no samples).

Keys (in the camera window):
    1-8   record the gesture shown in the printout (20 samples each)
    s     save to data/gestures.json and quit
'''

#----------------------------------------------------------
# Imports (standard library + third party)
#----------------------------------------------------------
import os
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_recognition import landmarks_to_vector, draw_skeleton


#----------------------------------------------------------
# Paths
#----------------------------------------------------------
# All paths are built from this file's location, so the project works wherever it is installed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "Dependencies"))   # Sits next to the project folder

HAND_MODEL_PATH = os.path.join(DEPENDENCIES_DIR, "Models", "hand_landmarker.task")  # MediaPipe hand model (not in the repo)
HAND_MODEL_DATA_PATH = os.path.join(BASE_DIR, "data", "gestures.json")              # Where the recorded samples are saved


#----------------------------------------------------------
# Constants
#----------------------------------------------------------
SAMPLES_PER_RECORDING = 20                  # Frames captured each time a recording key is pressed

KEY_TO_GESTURE = {                          # Keyboard key -> gesture name stored in gestures.json
    ord('1'): "turn_right",
    ord('2'): "turn_left",
    ord('3'): "move_right",
    ord('4'): "move_left",
    ord('5'): "move_forward_left_hand",
    ord('6'): "move_forward_right_hand",
    ord('7'): "move_backward_left_hand",
    ord('8'): "move_backward_right_hand",
}


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
# Load existing samples (or start with an empty set)
#----------------------------------------------------------
# Loading first means new recordings are added to the old ones instead of replacing them
if os.path.exists(HAND_MODEL_DATA_PATH):
    with open(HAND_MODEL_DATA_PATH) as f:
        registry = json.load(f)
else:
    registry = {
        "turn_right" : [],
        "turn_left" : [],
        "move_right" : [],
        "move_left" : [],
        "move_forward_left_hand" : [],
        "move_forward_right_hand" : [],
        "move_backward_left_hand" : [],
        "move_backward_right_hand" : []
        }


#----------------------------------------------------------
# Main loop
#----------------------------------------------------------
cap = cv2.VideoCapture(0)                   # Default webcam
frame_timestamp_ms = 0                      # MediaPipe video mode needs increasing timestamps (~30 fps)

print("\nPress a number key (1-8) to record a gesture, 's' to save & quit.\n")
print("Key  Gesture")
print("---  -------")
for k, name in KEY_TO_GESTURE.items():
    print(f"[{chr(k)}]  {name}")
print()

while cap.isOpened():
    # Grab a frame, mirror it (so it behaves like a mirror) and detect the hand
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # MediaPipe expects RGB, OpenCV gives BGR

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # Draw the skeleton and the instructions, then show the frame
    draw_skeleton(frame, result)

    cv2.putText(frame, "Press 1-8 to record, s to save+quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Recorder", frame)

    key = cv2.waitKey(1) & 0xFF             # Last key pressed (255 if none)

    #------------------------------------------------------
    # A recording key was pressed while a hand is visible
    #------------------------------------------------------
    if key in KEY_TO_GESTURE and result.hand_landmarks:
        gesture_name = KEY_TO_GESTURE[key]
        print(f"Recording {gesture_name}... hold the pose steady.")

        # Capture frames until enough samples with a visible hand have been collected
        collected = 0
        while collected < SAMPLES_PER_RECORDING:
            ok, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:                                       # Frames with no hand are skipped
                vector = landmarks_to_vector(result.hand_landmarks[0])      # Hand landmarks -> feature vector
                registry[gesture_name].append(vector.tolist())              # Lists, because numpy arrays can't be saved as JSON
                collected += 1

            draw_skeleton(frame, result)

            cv2.putText(frame, f"Capturing {gesture_name}: {collected}/{SAMPLES_PER_RECORDING}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Recorder", frame)
            cv2.waitKey(1)

        print(f"Done. {gesture_name} now has {len(registry[gesture_name])} total samples.")

    #------------------------------------------------------
    # 's' pressed: save everything and quit
    #------------------------------------------------------
    elif key == ord('s'):
        os.makedirs(os.path.dirname(HAND_MODEL_DATA_PATH), exist_ok=True)   # Create data/ if it doesn't exist yet
        with open(HAND_MODEL_DATA_PATH, "w") as f:
            json.dump(registry, f)
        print(f"Saved to {HAND_MODEL_DATA_PATH}")
        break


#----------------------------------------------------------
# Cleanup
#----------------------------------------------------------
cap.release()
cv2.destroyAllWindows()