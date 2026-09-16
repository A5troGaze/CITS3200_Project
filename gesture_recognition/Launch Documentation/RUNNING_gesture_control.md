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

## 3. Confirm the config.py file is configured correctly
Under `~/CITS3200/Dependencies/unitree_mujoco/simulate_python` open the `config.py` file.
Make sure:
- ROBOT = `"g1"`
- DOMAIN_ID = `1`
- INTERFACE = `"lo"`
- USE_JOYSTICK = `0`
- ENABLE_ELASTIC_BAND = `True`
Then **SAVE** the file

## 4. Navigate to and launch the mujoco simulation

```bash
cd ~/CITS3200/Dependencies/unitree_mujoco/simulate_python
```

```bash
python unitree_mujoco.py
```

A window should open displaying the mujoco simulation which will receive data via cyclonedds and unitree sdk

## 5. Navigate to and run the script
Open another terminal

```bash
cd ~/CITS3200/Project/gesture_recognition
```

```bash
python gesture_control.py [sim|real]
```

A window should open showing your webcam feed with green lines and red dots tracking your hand.

```
- Form a recorded gesture and take note of white string in top left corner of webcam feed and simulation robot's reaction to it
- Exit the simulation with the cross at the top right FIRST
- THEN press **`q`** (with the webcam window focused) to close the window cleanly.
- If it hangs, `Ctrl + C` in the terminal will force-stop it.
```

## 6. When you're done running it

```bash
deactivate
```

Exits the `g1-env` environment, returning to your normal shell.

---

**If the camera window doesn't open or shows an error:** this is very likely specific to your machine's camera setup (index, driver, or hardware), not a problem with the script itself. Check in with the team rather than assuming it's broken — camera detection quirks vary a lot machine to machine.
