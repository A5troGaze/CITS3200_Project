import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/home/ubuntu/CITS3200/models/pose_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1  # single person for now
)

cap = cv2.VideoCapture(0)
timestamp = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 1

        if result.pose_landmarks:
            h, w, _ = frame.shape
            for landmarks in result.pose_landmarks:
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        cv2.imshow("Pose Test - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
