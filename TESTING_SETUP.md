# Running and Testing the Code (Native Ubuntu Install)

This is a companion to `SETUP.md`, for anyone setting up a **fresh, native Ubuntu 22.04 install** (not a VM) to run and test the project code, e.g. the Person Identification camera/pose scripts. If you haven't done `SETUP.md`'s Steps 1-6 on this device yet, do that first, this document picks up after that and adds what's needed to actually run code against a real camera.

## Prerequisites checklist

Before starting here, confirm you have (from `SETUP.md`):
- Ubuntu 22.04 LTS installed and up to date (`sudo apt update && sudo apt upgrade -y`)
- Git + GitHub CLI installed and authenticated
- Python 3, pip, venv installed
- CycloneDDS built and `CYCLONEDDS_HOME` set
- The repo cloned to `~/CITS3200/Project`
- The `g1-env` virtual environment created with all 5 dependencies verified (`unitree_sdk2py`, `mediapipe`, `mujoco`, `pinocchio`, `general_motion_retargeting`)

If any of that is missing, follow `SETUP.md` first.

---

### Step 1: Install OpenCV

The pose/camera scripts use OpenCV for capturing and displaying the video feed. It isn't part of the 5 core dependencies verified in `SETUP.md`, so install it separately:

```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate   # Make sure the venv is active first
pip install opencv-python                            # Install OpenCV
python3 -c "import cv2; print('OpenCV OK', cv2.__version__)"   # Verify install
```

---

### Step 2: Confirm the camera is detected

On a native install, there's no VM passthrough step, Linux should see a built-in or USB webcam automatically. Confirm it:

```bash
ls /dev/video*
```

You should see at least one device, e.g. `/dev/video0`. If nothing shows up:
```bash
sudo apt install v4l-utils -y     # Installs video4linux utilities
v4l2-ctl --list-devices           # Lists detected camera hardware
```
If the camera still isn't listed, check it's not disabled in BIOS/UEFI (common on laptops with a physical camera privacy switch), and that no other application already has it locked.

---

### Step 3: Check camera permissions

Most Ubuntu Desktop installs add the default user to the `video` group automatically, but confirm it:
```bash
groups $USER
```
If `video` isn't listed:
```bash
sudo usermod -aG video $USER
```
Then log out and back in (group membership changes don't apply to an already-open session) before continuing.

---

### Step 4: Quick OpenCV camera test

Before touching MediaPipe, confirm you get a raw live feed:

```bash
nano ~/CITS3200/camera_test.py
```
Paste in:
```python
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera Test - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
Save and exit (`Ctrl+O`, Enter, `Ctrl+X`), then run:
```bash
python3 ~/CITS3200/camera_test.py
```
A window should open showing the live feed. Press `q` to close it.

---

### Step 5: Download the MediaPipe pose model

MediaPipe's Pose Landmarker needs a model file, it isn't bundled with the pip package:

```bash
mkdir -p ~/CITS3200/models
curl -o ~/CITS3200/models/pose_landmarker.task -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```
("lite" is the fastest/smallest model, good for testing; swap to "full" or "heavy" later if accuracy needs improving.)

---

### Step 6: Run the pose detection test

```bash
nano ~/CITS3200/pose_test.py
```
Paste in:
```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/home/YOUR_USERNAME/CITS3200/models/pose_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
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
```
**Replace `YOUR_USERNAME`** in the `model_asset_path` line with the actual username on this machine (check with `whoami` if unsure).

Save and exit, then run:
```bash
python3 ~/CITS3200/pose_test.py
```
You should see the live camera feed with green dots tracking body joints in real time.

---

### Step 7: Pull the latest project code

The scripts above are standalone tests, not part of the actual repo. Once camera + pose detection are confirmed working, get the real project code:
```bash
cd ~/CITS3200/Project
git pull
```
Check with the team which branch has the current Person Identification work if it's not on `main`, and check out accordingly:
```bash
git checkout <branch-name>
```

---

## Known gotcha (carried over from initial setup)

If you also need to (re)install **GMR** on this device as part of `SETUP.md`'s Step 5, install the CPU-only build of `torch` first, otherwise pip will pull several GB of CUDA-linked `nvidia-*` packages that are useless without GPU passthrough and can fill a modest disk:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd ~/CITS3200/Dependencies/GMR
pip install -e .
```

## Every new terminal session

Before running any project Python code:
```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```
