# Running `mediapipe_functionality_test.py` — Quick Start

## 1. Activate the shared environment

```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```

Your terminal prompt should now start with `(g1-env)`. This makes the `mediapipe`, `opencv-python`, and `numpy` packages (already installed in this shared venv) available.

## 2. Download the hand-tracking model file (first time only)

This is a large binary file MediaPipe needs at runtime. It lives outside the repo, alongside the other project dependencies, so it isn't included when you clone the repo — you need to download it once yourself:

```bash
mkdir -p ~/CITS3200/Dependencies/Models
curl -o ~/CITS3200/Dependencies/Models/hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Confirm it downloaded correctly (should be several MB, not 0 bytes):

```bash
ls -la ~/CITS3200/Dependencies/Models/hand_landmarker.task
```

You only need to do this once — the file stays on your machine after that.

## 3. Navigate to the script

```bash
cd ~/CITS3200/Project/gesture_recognition
cd "Functionality Tests"
```

## 4. Run the script

```bash
python mediapipe_functionality_test.py
```

A window should open showing your webcam feed with green lines and red dots tracking your hand.

- Press **`q`** (with the window focused) to close it cleanly.
- If it hangs, `Ctrl + C` in the terminal will force-stop it.

## 5. When you're done running it

```bash
deactivate
```

Exits the `g1-env` environment, returning to your normal shell.

---

**If the camera window doesn't open or shows an error:** this is very likely specific to your machine's camera setup (index, driver, or hardware), not a problem with the script itself. Check in with the team rather than assuming it's broken — camera detection quirks vary a lot machine to machine.
