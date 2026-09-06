'''Control mujoco model with registered hand signals'''

#== Import Dependencies ==============================================================
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
# from unitree_sdk2py.core.channel import ChannelFactoryInitialize
# from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from gesture_core import landmarks_to_vector, classify, draw_skeleton, HAND_CONNECTIONS

'''
#== Define SDK Parameters ===========================================================
ChannelFactoryInitialize(0, "lo")
sport_client = LocoClient()
sport_client.SetTimeout(10.0)
sport_client.Init()
'''


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
stream = cv2.VideoCapture(0)
frame_timestamp_ms = 0

display_enabled = True
last_gesture = None
best_overall_dist = float("inf")


#== Load G1 Model ====================================================================
model = mj.MjModel.from_xml_path(G1_SCENE_PATH)
data = mj.MjData(model)


#== Find Actuator to target function ==
def find_actuator(model, keyword):
    for i in range(model.nu):                                       # For each actuator that the model (G1) possesses
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if name and keyword in name.lower():                        # If the actuator name and the keyword match
            return i                                                # Return the index of the actuator
    return None                                                     # Else return None if matching actuator cannot be found



GESTURE_ACTIONS = {
    "turn_right":               ("hip_yaw",        -0.3),
    "turn_left":                ("hip_yaw",         0.3),
    "move_right":               ("hip_roll",       -0.2),
    "move_left":                ("hip_roll",        0.2),
    "move_forward_left_hand":   ("left_shoulder",   0.5),
    "move_forward_right_hand":  ("right_shoulder",  0.5),
    "move_backward_left_hand":  ("left_shoulder",  -0.3),
    "move_backward_right_hand": ("right_shoulder", -0.3)
}


#== 
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



def main():
    global frame_timestamp_ms, last_gesture, best_overall_dist, display_enabled


    with mj.viewer.launch_passive(model, data) as viewer:
        while stream.isOpened() and viewer.is_running():
            ok, frame = stream.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33
            result = gesture_model.detect_for_video(mp_image, frame_timestamp_ms)

            #== Recognise gesture
            gesture_name = None
            if result.hand_landmarks:
                vector = landmarks_to_vector(result.hand_landmarks[0])
                gesture_name, best_overall_dist = classify(vector, registry)

            '''
            if gesture_name == "turn_right":
                sport_client.Move(0, 0, -0.3)
            elif gesture_name == "turn_left":
                sport_client.Move(0, 0, 0.3)
            elif gesture_name == "move_right":
                sport_client.Move(0, -0.3, 0)
            elif gesture_name == "move_left":
                sport_client.Move(0, -0.3, 0)
            elif gesture_name == "move_backward_right_hand":
                sport_client.Move(-0.3, 0, 0)
            elif gesture_name == "move_backward_left_hand":
                sport_client.Move(-0.3, 0, 0)
            elif gesture_name == "move_forward_right_hand":
                sport_client.Move(0.3, 0, 0)
            elif gesture_name == "move_forward_left_hand":
                sport_client.Move(0.3, 0, 0)'''
            
            #== React to gesture
            if gesture_name is not None and gesture_name != last_gesture:
                print(f"Gesture: {gesture_name} (dist={best_overall_dist:.2f})")
                apply_gesture(model, data, gesture_name)
                last_gesture = gesture_name
            elif gesture_name is None:
                last_gesture = None

            #==
            mj.mj_step(model, data)
            viewer.sync()

            #==
            if display_enabled:
                draw_skeleton(frame, result)
                label = gesture_name or "no match"
                cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.imshow("Gesture Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                display_enabled = not display_enabled
                if not display_enabled:
                    cv2.destroyWindow("Gesture Control")

    stream.release()
    cv2.destroyAllWindows()



#== Run ==
if __name__ == "__main__":
    main()