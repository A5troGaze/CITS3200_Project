import cv2
import json
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gesture_core import landmarks_to_vector

MODEL_PATH = os.path.expanduser("~/CITS3200/Dependencies/Models/hand_landmarker.task")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "gestures.json")

#== Set up landmarker model ====================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)  # Set path to model
options = vision.HandLandmarkerOptions(                         # Configure model:
    base_options=base_options,                                      # Path
    num_hands=1,                                                    # Number of hands
    running_mode=vision.RunningMode.VIDEO                           # Expect video stream
)
landmarker = vision.HandLandmarker.create_from_options(options) # Create the useable object

if os.path.exists(DATA_PATH):
    with open(DATA_PATH) as f:
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

SAMPLES_PER_RECORDING = 20

KEY_TO_GESTURE = {
    ord('1'): "turn_right",
    ord('2'): "turn_left",
    ord('3'): "move_right",
    ord('4'): "move_left",
    ord('5'): "move_forward_left_hand",
    ord('6'): "move_forward_right_hand",
    ord('7'): "move_backward_left_hand",
    ord('8'): "move_backward_right_hand",
}

#== Define hand model ==========================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17) 
]

#== Draw 21-point hand skeleton ================================================
def draw_skeleton(frame, result):
    if not result.hand_landmarks:
        return
    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)


cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

print("Press 1-8 to record a gesture, 's' to save & quit.")
print("Gestures: " + ", ".join(f"{chr(k)}={v}" for k, v in KEY_TO_GESTURE.items()))

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    draw_skeleton(frame, result)

    cv2.putText(frame, "Press 1-8 to record, s to save+quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Recorder", frame)

    key = cv2.waitKey(1) & 0xFF




    #==  ========
    if key in KEY_TO_GESTURE and result.hand_landmarks:
        gesture_name = KEY_TO_GESTURE[key]
        print(f"Recording {gesture_name}... hold the pose steady.")

        collected = 0
        while collected < SAMPLES_PER_RECORDING:
            ok, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                vector = landmarks_to_vector(result.hand_landmarks[0])
                registry[gesture_name].append(vector.tolist())
                collected += 1

            draw_skeleton(frame, result)

            cv2.putText(frame, f"Capturing {gesture_name}: {collected}/{SAMPLES_PER_RECORDING}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Recorder", frame)
            cv2.waitKey(1)

        print(f"Done. {gesture_name} now has {len(registry[gesture_name])} total samples.")

    elif key == ord('s'):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, "w") as f:
            json.dump(registry, f)
        print(f"Saved to {DATA_PATH}")
        break

cap.release()
cv2.destroyAllWindows()