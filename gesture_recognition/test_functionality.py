# Test Functionality of dependencies being used in gesture recognition product

#== Import all dependencies for testing ========================
import cv2                                              # OpenCV. Handles camera
import os                                               # os
import mediapipe as mp                                  # Core Mediapipe library
from mediapipe.tasks import python                      # Mediapipe's Tasks API
from mediapipe.tasks.python import vision               # Mediapipe's Tasks vision aspect

#== Path to the hand_landmarker model used for gesture recognition =====================================
MODEL_PATH = os.path.expanduser("~/CITS3200/Dependencies/Models/hand_landmarker.task")

#== Define hand model ==========================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17) 
]

#== Set up landmarker model ====================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)  # Set path to model
options = vision.HandLandmarkerOptions(                         # Configure model:
    base_options=base_options,                                      # Path
    num_hands=1,                                                    # Number of hands
    running_mode=vision.RunningMode.VIDEO                           # Expect video stream
)
landmarker = vision.HandLandmarker.create_from_options(options) # Create the useable object

#== Camera Setup ===============================================================
cap = cv2.VideoCapture(0)   # Opens device at index 0 (usually webcam, change index if another device is required)
frame_timestamp_ms = 0      # Running counter that gets incremented after every frame

#== Main Loop ==================================================================
while cap.isOpened():
    #== Run while camera is open ======
    ok, frame = cap.read()
    if not ok:
        break
    #==================================

    
    frame = cv2.flip(frame, 1)                                                          # Mirror image from camera so it is unflipped
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                  # Convert OpenCV video stream from BGR to RGB for Mediapipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)               # Convert RGB array to MediaPipe's expected format

    frame_timestamp_ms += 33                                                            # Increment timestamp by 33ms (given 30fps [1000ms/30 ~=33])
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)                  # Return MediaPipe findings (a list of detected hands [21 landmark points])

    #== Draw results ==================
    if result.hand_landmarks:
        h, w, _ = frame.shape                                                           # Get height and width (discard channel count) from frame
        for hand_landmarks in result.hand_landmarks:                                    # For each hand (Only 1 currently):

            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]              # Get real pixel position

            for start_indx, end_indx in HAND_CONNECTIONS:                                   # For each connected pair in defined hand model
                cv2.line(frame, points[start_indx], points[end_indx], (0, 255, 0), 2)       # Draw a line from the first to the second point

            for point in points:                                                            # Likewise:
                cv2.circle(frame, point, 4, (0, 255, 0), -1)                                # draw a circle at each point
    # =================================

    #== Display in a frame ============
    cv2.imshow("Landmark Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Wait for keypress and break loop if 'q' is pressed
        break

cap.release()               # Free camera device so other programs can use it again
cv2.destroyAllWindows       # Closes open CV window.