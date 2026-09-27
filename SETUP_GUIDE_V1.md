# CITS3200 G1 Project — Full Setup Guide

This covers everything needed to reproduce the current project on a fresh Ubuntu 22.04 machine: the gesture-recognition pipeline, the MuJoCo sim, the C++ walking controller (`g1_ctrl` from `unitree_rl_lab`), and the virtual-gamepad bridge that connects gestures to it. Follow it in order — later steps depend on earlier ones.

Assumed OS: Ubuntu 22.04 (native, not WSL/Docker — CycloneDDS needs direct network interface access, which containers on Windows/Mac can't reliably provide).

---

## 0. Directory layout

Everything third-party lives outside the git repo, in `~/HumanoidControl/Dependencies/`. Only your own code lives in `~/HumanoidControl/Project/`. This keeps the repo clean and matches what the rest of this guide assumes.

Here's what you'll end up with once every step below is done — use this as a map to check your progress against:

```
~
├── cyclonedds/
│   └── install/                        ← CycloneDDS 0.10.2, built from source (§3)
│
└── HumanoidControl/
    ├── Dependencies/                   ← everything third-party, outside the repo
    │   ├── g1-env/                     ← Python virtual environment (§2)
    │   ├── unitree_sdk2_python/        ← Python DDS SDK, editable install (§4)
    │   ├── unitree_sdk2/               ← C++ DDS SDK (§8)
    │   │   └── build/                  ← installs to /usr/local, not run from here
    │   ├── Models/
    │   │   └── hand_landmarker.task    ← MediaPipe hand model (§5)
    │   ├── unitree_mujoco/             ← DDS-aware simulator (§9)
    │   │   └── simulate/               ← C++ build this project runs
    │   │       ├── config.yaml         ← domain_id / use_joystick / enable_elastic_band
    │   │       └── build/
    │   │           └── unitree_mujoco  ← compiled simulator binary (Terminal 2)
    │   └── unitree_rl_lab/             ← walking policy + controller (§10)
    │       └── deploy/
    │           ├── thirdparty/
    │           │   └── onnxruntime-linux-x64-1.22.0/
    │           └── robots/
    │               └── g1_29dof/
    │                   ├── config/
    │                   │   └── config.yaml
    │                   └── build/
    │                       └── g1_ctrl         ← compiled walking controller (Terminal 3)
    │
    └── Project/                        ← the actual git repo (github.com/A5troGaze/CITS3200_Project)
        ├── .pytest_cache/
        ├── Documentation/
        ├── gesture_library/
        ├── gesture_recognition/
        │   ├── __pycache__/
        │   ├── data/
        │   │   └── gestures.json        ← recorded gesture reference samples
        │   ├── Debugging Tools/
        │   ├── Functionality Tests/
        │   ├── Launch Documentation/
        │   ├── Testing Documentation/
        │   ├── person_id/
        │   ├── abstract_controller.py      ← shared controller interface
        │   ├── gesture_control.py          ← camera loop wiring recognition to a controller
        │   ├── gesture_core.py             ← landmark normalization + classification
        │   ├── gesture_to_vgamepad.py      ← virtual-gamepad bridge into g1_ctrl (Terminal 1, §12)
        │   ├── react_to_gestures.py        ← dispatches recognized gestures to a controller
        │   ├── real_controller.py          ← drives the physical robot
        │   ├── recognize_gestures.py       ← live recognition with debounce/threshold
        │   ├── record_gestures.py          ← records reference samples per gesture
        │   └── simulation_controller.py    ← drives the MuJoCo sim
        ├── .gitignore
        └── SETUP.md
```

```bash
mkdir -p ~/HumanoidControl/Dependencies
```

By the end of this guide, `~/HumanoidControl/Dependencies/` will contain: `g1-env/` (the Python venv), `unitree_sdk2_python/`, `unitree_sdk2/` (C++ SDK), `unitree_mujoco/`, `unitree_rl_lab/`, and `Models/` (for the hand-landmark model file). `~/HumanoidControl/Project/` is your actual git clone of the team repo.

```bash
cd ~/HumanoidControl && git clone https://github.com/A5troGaze/CITS3200_Project.git Project
```

---

## 1. System packages

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake git python3-venv python3-dev \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev \
  xdotool cheese
```

`build-essential`/`cmake` are needed to compile `unitree_sdk2`, `unitree_mujoco`'s C++ binary, and `g1_ctrl`. The `lib*-dev` packages are direct dependencies of `g1_ctrl`'s build (yaml-cpp for config files, Boost for its command-line parsing, Eigen for math, spdlog/fmt for logging). `xdotool` sends keyboard events into the MuJoCo viewer window programmatically (used for the elastic-band controls). `cheese` is a webcam viewer — on this hardware it's needed once at the start of each session to prime the camera into the right pixel format (see the camera section below).

---

## 2. Python virtual environment

```bash
python3 -m venv ~/HumanoidControl/Dependencies/g1-env
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
```

Everything Python-related in this guide assumes this venv is active — always `source` it before running any of the commands below in a new terminal. Note: a venv bakes in its own absolute path; if you ever move `~/HumanoidControl/Dependencies/g1-env/` to a different location, it breaks and must be recreated, not just moved.

```bash
pip install --upgrade pip
pip install mujoco numpy opencv-python mediapipe vgamepad unitree_sdk2py
```

`mujoco` is the physics engine itself (plain `pip install mujoco` has no networking/DDS support — see the CycloneDDS section for why that matters). `opencv-python` and `mediapipe` are the camera/hand-tracking stack for gesture recognition. `vgamepad` creates a virtual Xbox360 controller at the OS level, used to drive the walking controller's joystick input. `unitree_sdk2py` is the Python DDS SDK — it needs CycloneDDS underneath it, set up next.

---

## 3. CycloneDDS (built from source)

The DDS middleware `unitree_sdk2py` and `unitree_sdk2` (C++) both sit on top of. Built from source rather than pip-installed for version control (this project pins tag `0.10.2`).

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git ~/cyclonedds
cd ~/cyclonedds && git checkout 0.10.2
mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

This clones CycloneDDS to `~/cyclonedds` (deliberately outside `~/HumanoidControl/`, since it's a system-level dependency rather than a project-specific one), checks out the pinned version, and builds it with its install prefix pointed at `~/cyclonedds/install` rather than a system location.

```bash
echo 'export CYCLONEDDS_HOME=~/cyclonedds/install' >> ~/.bashrc
source ~/.bashrc
```

`unitree_sdk2_python`'s build looks for CycloneDDS via this environment variable — without it, its own build step below will fail to find the DDS libraries.

---

## 4. unitree_sdk2_python (editable install)

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

`-e` (editable) installs it in-place rather than copying it into site-packages, so any local edits to the SDK take effect immediately without reinstalling.

---

## 5. Hand-landmark model

```bash
mkdir -p ~/HumanoidControl/Dependencies/Models
```

Download `hand_landmarker.task` from MediaPipe's model zoo into that folder (check `SETUP.md` in the repo for the exact download link/version currently in use — this file is kept outside the repo like everything else in `Dependencies/`).

---

## 6. Camera setup

This project uses an Insta360 One RS in Webcam Mode over USB (a workaround for a non-functional internal webcam on this specific machine — skip this step if your machine's built-in camera works fine).

Known gotcha: OpenCV defaults to the wrong pixel format on a cold connection to this camera, producing corrupted color. Fix: open Cheese (a GStreamer-based webcam app) once first — it primes the camera into MJPG mode, which OpenCV then inherits correctly for the rest of the session.

```bash
cheese
```

Open it, confirm you see a working camera preview, then close it. Any script using OpenCV afterward in that session should now get correct color.

---

## 7. Gesture recognition pipeline

This part is just your own repo code — no separate install beyond what's already in the venv (`opencv-python`, `mediapipe`). Relevant files in `Project/gesture_recognition/`:

- `record_gestures.py` — records reference samples for each gesture (8 gestures currently: move left/right, turn left/right, move forward/backward with left or right hand), keys 1–8
- `gesture_core.py` — landmark normalization and nearest-neighbor classification against recorded samples
- `recognize_gestures.py` — live recognition loop with a distance threshold (currently tuned to `1.5`) deciding whether a detected hand pose counts as a real gesture command or noise

Run recognition standalone to confirm the camera + MediaPipe + classifier chain works before wiring it into robot control:

```bash
cd ~/HumanoidControl/Project/gesture_recognition
python recognize_gestures.py
```

---

## 8. unitree_sdk2 (C++ SDK)

A separate C++ build from `unitree_sdk2_python` — this is what `g1_ctrl` and the C++ MuJoCo simulator link against.

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install
sudo ldconfig
```

`-DBUILD_EXAMPLES=OFF` skips compiling the SDK's own example programs, which aren't needed here. `sudo make install` copies headers to `/usr/local/include/unitree/` and libraries to `/usr/local/lib` (both `g1_ctrl` and `unitree_mujoco`'s C++ build hardcode `/usr/local` paths rather than using `find_package`, so this exact install location matters). `sudo ldconfig` refreshes the dynamic linker's cache — without this, the linker cache can go stale and later binaries fail with "library not found" errors even though the files are genuinely present.

Verify:

```bash
ls -la /usr/local/lib/libddsc.so* /usr/local/lib/libddscxx.so*
```

Both should list real files, not "No such file."

---

## 9. unitree_mujoco (the DDS-aware simulator)

This wraps plain MuJoCo (already installed in step 2) with CycloneDDS publish/subscribe, replacing direct `data.ctrl` writes with the same `LowState_`/`LowCmd_` DDS messages the real robot uses — meaning control code written against this interface runs unmodified on real hardware later.

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

This repo also ships a pure-Python variant (`simulate_python/`), but this project doesn't use it — a Python-vs-C++ CycloneDDS mismatch caused real connection problems with `g1_ctrl` during development, so only the C++ `simulate/` build below is set up.

Build it:

```bash
cd ~/HumanoidControl/Dependencies/unitree_mujoco/simulate
mkdir build && cd build
cmake .. && make
```

Configure it — edit `~/HumanoidControl/Dependencies/unitree_mujoco/simulate/config.yaml`:

```yaml
robot: "g1"
domain_id: 0
use_joystick: 1
joystick_type: "xbox"
enable_elastic_band: 1
```

`domain_id: 0` is not optional — `g1_ctrl`'s DDS domain is hardcoded to `0` in its own source with no way to configure it, so this simulator must match or the two processes will never discover each other over DDS (they'll each start fine, publish/subscribe fine, and just never see each other — a very quiet failure mode with no error message). `use_joystick: 1` and `enable_elastic_band: 1` are both required for the gesture-bridge workflow in the next section — without the elastic band, the robot has nothing holding it up while `g1_ctrl` transitions it into a standing pose.

Sanity-check it runs:

```bash
cd ~/HumanoidControl/Dependencies/unitree_mujoco/simulate/build
./unitree_mujoco
```

Should print the model's link/joint/actuator/sensor listing and open a viewer window.

---

## 10. unitree_rl_lab (the walking controller, `g1_ctrl`)

This is the actual trained walking policy — an ONNX neural network wrapped in a C++ program (`g1_ctrl`) that reads joystick-style commands over DDS and outputs joint torques.

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
```

It needs ONNX Runtime's C++ distribution, which isn't fetched automatically — download it manually into the location its `CMakeLists.txt` expects:

```bash
cd ~/HumanoidControl/Dependencies/unitree_rl_lab/deploy/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz
```

Build the G1-29dof controller:

```bash
cd ~/HumanoidControl/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make
```

This links against the `unitree_sdk2` install from step 8 (via its hardcoded `/usr/local` paths) and the ONNX Runtime you just downloaded, producing a `g1_ctrl` executable in this `build/` folder.

Sanity-check it (with `unitree_mujoco` from step 9 already running in another terminal):

```bash
./g1_ctrl --network lo
```

`--network lo` pins its DDS participant to the loopback interface explicitly — without it, `g1_ctrl` defaults to an empty interface string, which can cause it to auto-select a different network interface than the simulator is using, so the two never connect even on the same domain. Watch for `Connected to robot.` — that's confirmation it found `unitree_mujoco`'s published state.

---

## 11. vgamepad / uinput permissions

The gesture-to-robot bridge drives a virtual Xbox360 controller (via `vgamepad`, installed in step 2) rather than publishing DDS messages directly, so that `unitree_mujoco`'s own already-correct joystick-reading code does the translation.

```bash
sudo usermod -aG input $USER
```

then log out and back in (group membership changes don't take effect in an already-open session). Alternatively, for a one-off without logging out:

```bash
sudo chmod +0666 /dev/uinput
```

This grants access to `/dev/uinput`, the kernel interface `vgamepad` uses to create the virtual device. The permanent fix is a udev rule:

```bash
echo 'KERNEL=="uinput", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/50-uinput.rules
sudo udevadm control --reload-rules
```

Verify the virtual gamepad actually works before relying on it:

```bash
python3 -c "import vgamepad as vg; gp = vg.VX360Gamepad(); print('OK')"
```

If this raises a permissions error, the `uinput` access above didn't take effect — recheck group membership or the udev rule.

---

## 12. Running the full pipeline

Four terminals, all with `g1-env` activated (`source ~/HumanoidControl/Dependencies/g1-env/bin/activate`), started in this order:

**Terminal 1** — the gesture-to-gamepad bridge (creates the virtual controller; must start before the simulator, which checks for a joystick at startup):

```bash
cd ~/HumanoidControl/Project/gesture_recognition
python gesture_to_vgamepad.py
```

It'll pause and wait for input before actually driving anything — leave it there for now.

**Terminal 2** — the simulator:

```bash
cd ~/HumanoidControl/Dependencies/unitree_mujoco/simulate/build
./unitree_mujoco
```

**Terminal 3** — the walking controller:

```bash
cd ~/HumanoidControl/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

Confirm it prints `Connected to robot.` before continuing.

**Back in Terminal 1**: press Enter to run the automated stand-up sequence (holds the trigger to enter FixStand mode, prompts you to loosen the elastic band until the feet touch the ground, starts the walking policy, then releases the band). Once it says gestures are streaming, either run `recognize_gestures.py`'s output into it, or (for a standalone sanity check with no camera) type gesture names directly at its prompt.

---

## Known gotchas

- A Python venv bakes in its absolute creation path — moving the `Dependencies/` folder breaks `g1-env` and requires recreating it.
- The Insta360 camera needs Cheese opened once per session before OpenCV, or colors come out corrupted.
- `g1_ctrl`'s DDS domain is hardcoded to `0` in its source and cannot be changed via config — the simulator's `domain_id` must match this exactly.
- After any `sudo make install` of a C++ library, run `sudo ldconfig` — otherwise the linker cache can be stale and dependent binaries fail to find libraries that are genuinely present.
- The joystick's trigger axes (`LT`/`RT`) and analog sticks all pass through a slow smoothing filter inside `unitree_sdk2`'s joystick code (roughly a 3%-per-tick exponential ramp) — a command needs to be held for real time (not just an instant) to actually register, and a rising-edge condition combined with a trigger (like the `L2 + Up` stand-up combo) needs the trigger pressed and settled *before* the button, not simultaneously.
- The MuJoCo elastic band's slack only changes by 0.1m per single keypress of `8`/`7` — reaching the ground from the default hoisted position needs many repeated presses, sent incrementally while watching the sim, not a single press.
- CycloneDDS requires direct network interface binding — this is why native Ubuntu is used rather than Docker on Windows/Mac for anything involving the SDK.
