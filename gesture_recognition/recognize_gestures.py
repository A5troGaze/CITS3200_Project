# Real-time gesture recognition: reads video (camera or file), detects hand
# landmarks, and prints the recognized gesture name to console whenever it changes.

import argparse
import cv2
import json
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gesture_core import landmarks_to_vector, classify

MODEL_PATH = os.path.expanduser("~/CITS3200/Dependencies/Models/hand_landmarker.task")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "gestures.json")

#== Command line args ===========================================================
# Default: use the live camera (index 0), same as the rest of the team's scripts.
# Pass --video <path> to test against a pre-recorded file instead (useful if your
# machine/VM doesn't have camera access — the rest of the script behaves identically).
parser = argparse.ArgumentParser()
parser.add_argument("--video", default=None, help="Path to a video file to use instead of the live camera")
args = parser.parse_args()

#== Load the recorded gesture registry ==========================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Couldn't find {DATA_PATH}. Make sure data/gestures.json (recorded with "
        "record_gestures.py) is present before running recognition."
    )

with open(DATA_PATH) as f:
    registry = json.load(f)

#== Set up landmarker model ======================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)  # Set path to model
options = vision.HandLandmarkerOptions(                         # Configure model:
    base_options=base_options,                                      # Path
    num_hands=1,                                                    # Number of hands
    running_mode=vision.RunningMode.VIDEO                           # Expect video stream
)
landmarker = vision.HandLandmarker.create_from_options(options)  # Create the useable object

#== Camera / video setup =========================================================
cap = cv2.VideoCapture(args.video if args.video else 0)
frame_timestamp_ms = 0

#== Display is optional: some machines (e.g. headless VMs) can't open a GUI window.
# On Linux, no DISPLAY env var means there's no X11/GUI available at all — calling
# cv2.imshow() in that case crashes the whole process (not a catchable exception),
# so we check for this BEFORE ever attempting to open a window, rather than after.
display_enabled = True
if os.name == "posix" and "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    display_enabled = False
    print("(No display detected — running in console-only mode.)")

last_gesture = None  # Tracks the previous frame's result so we only print on change
best_overall_dist = float("inf")  # Tracks the smallest distance seen in the whole run, even on frames that never beat the threshold

print("Recognizing gestures from " + ("camera (index 0)" if not args.video else f"video file: {args.video}"))
print("Press 'q' to quit (if a display window is open), or Ctrl+C in the terminal.")

#== Main loop =====================================================================
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    #== Classify whatever hand was found this frame =============================
    if result.hand_landmarks:
        vector = landmarks_to_vector(result.hand_landmarks[0])
        name, dist = classify(vector, registry)
        current_gesture = name if name else "no match"
        best_overall_dist = min(best_overall_dist, dist)  # Track how close this frame got, matched or not
    else:
        current_gesture = "no hand"
        dist = None

    #== TEMPORARY DEBUG: print distance a few times a second, regardless of change
    if dist is not None and frame_timestamp_ms % 300 < 33:
        print(f"  live: {current_gesture} dist={dist:.3f}")

    #== Only print when the recognized gesture actually changes =================
    if current_gesture != last_gesture:
        if dist is not None:
            print(f"[{frame_timestamp_ms}ms] {current_gesture} (distance={dist:.3f})")
        else:
            print(f"[{frame_timestamp_ms}ms] {current_gesture}")
        last_gesture = current_gesture

    #== Optional display: label the frame and show it, if a display is available
    if display_enabled:
        label = current_gesture if current_gesture not in ("no hand",) else "..."
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        try:
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # No GUI available (e.g. headless VM) — keep running console-only.
            print("(No display available — continuing in console-only mode.)")
            display_enabled = False

#== Summary: how close did we ever get, even if nothing beat the threshold? ======
if best_overall_dist < float("inf"):
    print(f"Closest distance seen across the whole run: {best_overall_dist:.3f} (threshold is 0.3)")
else:
    print("No hand was ever detected in this run.")

cap.release()
landmarker.close()
if display_enabled:
    cv2.destroyAllWindows()
