# Running `react_to_gestures.py` — Quick Start

## 1. Activate the shared environment

```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```

Your terminal prompt should now start with `(g1-env)`. This makes the `mediapipe`, `opencv-python`, and `numpy` packages (already installed in this shared venv) available.

## 2. Confirm the hand-tracking model file is present

Confirm the model file is downloaded correctly (should be several MB, not 0 bytes):

```bash
ls -la ~/CITS3200/Dependencies/Models/hand_landmarker.task
```

Confirm the mujoco menagerie model and scene file is downloaded correctly:
```bash
ls -la ~/CITS3200/Dependencies/mujoco_menagerie/unitree_g1/g1_with_hands.xml        # Check model file
ls -la ~/CITS3200/Dependencies/mujoco_menagerie/unitree_g1/scene_with_hands.xml     # Check scene file
```


## 3. Navigate to the script

```bash
cd ~/CITS3200/Project/gesture_recognition
```

## 4. Run the script

```bash
python react_to_gestures.py
```

A window should open showing your webcam feed with green lines and red dots tracking your hand.
Another window should also open displaying the mujoco simulation

```
- Form a recorded gesture and take note of green letters in top left corner of webcam feed and simulation robot's reaction to it
- Press **`q`** (with the window focused) to close the window cleanly.
- If it hangs, `Ctrl + C` in the terminal will force-stop it.```

## 5. When you're done running it

```bash
deactivate
```

Exits the `g1-env` environment, returning to your normal shell.

---

**If the camera window doesn't open or shows an error:** this is very likely specific to your machine's camera setup (index, driver, or hardware), not a problem with the script itself. Check in with the team rather than assuming it's broken — camera detection quirks vary a lot machine to machine.
