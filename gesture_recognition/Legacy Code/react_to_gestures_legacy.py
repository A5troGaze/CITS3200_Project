'''Control mujoco model with registered hand signals'''

#== Import Dependencies ==============================================================
import argparse
import json
import os
import sys
import time
import cv2

import numpy as np
import mujoco as mj
import mujoco.viewer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_core import landmarks_to_vector, classify, draw_skeleton, HAND_CONNECTIONS


#== Define Paths =====================================================================
HAND_MODEL_PATH = os.path.expanduser("~/CITS3200/Dependencies/Models/hand_landmarker.task")
HAND_MODEL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "gestures.json")
G1_SCENE_PATH = os.path.expanduser("~/CITS3200/Dependencies/mujoco_menagerie/unitree_g1/scene_with_hands.xml")

for i in [HAND_MODEL_DATA_PATH, HAND_MODEL_PATH, G1_SCENE_PATH]:    # For each path
    if not os.path.exists(i):                                       # If path does not exist; Raise Error
        raise FileNotFoundError(
            f"Couldn't find {i}. Make sure {i} is present before running again."
        )


#== Load Gestures ====================================================================
with open(HAND_MODEL_DATA_PATH) as saved_gestures:                  # Load saved gestures into a registry
    registry = json.load(saved_gestures)


#== Set Up Hand Model ================================================================
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(                             # Define Model:
    base_options = base_options,                                    # Path
    num_hands = 1,                                                  # Number of hands for gestures
    running_mode = vision.RunningMode.VIDEO                         # Video Stream
)
gesture_model = vision.HandLandmarker.create_from_options(options)  # Create useable object from definition


#== Set Up Camera / Video ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default=None,
    help="Path to a video file to use instead of the live camera"
)
args = parser.parse_args()

stream = cv2.VideoCapture(args.video if args.video else 0)
frame_timestamp_ms = 0

display_enabled = True
last_gesture = None
best_overall_dist = float("inf")



#== Load G1 Model ====================================================================
model = mj.MjModel.from_xml_path(G1_SCENE_PATH)
data = mj.MjData(model)


#== Find Actuator to target function =================================================
def find_actuator(model, keyword):
    for i in range(model.nu):                                       # For each actuator that the model (G1) possesses
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if name and keyword in name.lower():                        # If the actuator name and the keyword match
            return i                                                # Return the index of the actuator
    return None                                                     # Else return None if matching actuator cannot be found


#== Defined actions for testing signal reaction ======================================
GESTURE_ACTIONS = {
    "turn_right":               ("right_hip_yaw_joint",         1.5),
    "turn_left":                ("left_hip_yaw_joint",         -1.5),
    "move_right":               ("right_hip_roll_joint",        1.5),
    "move_left":                ("left_hip_roll_joint",        -1.5),
    "move_forward_left_hand":   ("left_shoulder_pitch_joint",  -0.5),
    "move_forward_right_hand":  ("right_shoulder_pitch_joint", -0.5),
    "move_backward_left_hand":  ("left_shoulder_pitch_joint",   0.5),
    "move_backward_right_hand": ("right_shoulder_pitch_joint",  0.5)
}


#== Apply the gesture's action to the simulated model ================================
def apply_gesture(model, data, gesture_name):
    action = GESTURE_ACTIONS.get(gesture_name)
    if action is None:
        return
    keyword, target = action
    actuator_id = find_actuator(model, keyword)
    if actuator_id is None:
        print(f"[WARNING] no actuator matching '{keyword}' for gesture '{gesture_name}'")
        return
    data.ctrl[actuator_id] = target


#== main app loop ===================================================================
def main():
    global frame_timestamp_ms, last_gesture, best_overall_dist, display_enabled


    with mj.viewer.launch_passive(model, data) as viewer:                           # Open mujoco
        while stream.isOpened() and viewer.is_running():                            # Using the camera stream
            ok, frame = stream.read()
            if not ok:
                break

            #== Prep intake from camera =============================================
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33
            result = gesture_model.detect_for_video(mp_image, frame_timestamp_ms)

            #== Recognise gesture ===================================================
            gesture_name = None
            if result.hand_landmarks:
                vector = landmarks_to_vector(result.hand_landmarks[0])
                gesture_name, best_overall_dist = classify(vector, registry)
            
            #== React to gesture ====================================================
            if gesture_name is not None and gesture_name != last_gesture:
                print(f"Gesture: {gesture_name} | dist={best_overall_dist:.2f} | {frame_timestamp_ms}ms")
                apply_gesture(model, data, gesture_name)
                last_gesture = gesture_name
            elif gesture_name is None:
                last_gesture = None

            #== Step the model to next move ========================================
            mj.mj_step(model, data)
            viewer.sync()

            #== Display gesture skeleton outline
            if display_enabled:
                draw_skeleton(frame, result)
                label = gesture_name or "no match"
                cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.imshow("Gesture Control", frame)

            #== Exit and dis/enable display ========================================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                display_enabled = not display_enabled
                if not display_enabled:
                    cv2.destroyWindow("Gesture Control")

    stream.release()
    cv2.destroyAllWindows()



#== Run ============================================================================
if __name__ == "__main__":
    main()
