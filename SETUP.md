# Humanoid Control (Unitree G1): Setup and Run Guide

This guide takes a fresh Ubuntu 22.04 machine to a working copy of the project: a webcam watches the operator, hand gestures make the Unitree G1 walk, and mimic mode makes the robot's arms copy the operator's arms. Everything runs in the Unitree MuJoCo simulator first; the same program can then drive the physical G1.

Follow the steps in order. Later steps depend on earlier ones. Expect 1 to 2 hours, most of it compiling and downloading.

---

## Contents

1. [What you need](#1-what-you-need)
2. [How the system fits together](#2-how-the-system-fits-together)
3. [Folder layout](#3-folder-layout)
4. [System packages](#4-system-packages)
5. [Get the project code](#5-get-the-project-code)
6. [CycloneDDS](#6-cyclonedds-0102)
7. [Python environment](#7-python-environment)
8. [MediaPipe models](#8-mediapipe-models)
9. [Unitree C++ SDK](#9-unitree-c-sdk-unitree_sdk2)
10. [MuJoCo simulator (patched)](#10-mujoco-simulator-unitree_mujoco-patched)
11. [Walking controller (g1_ctrl)](#11-walking-controller-g1_ctrl)
12. [Virtual gamepad permissions](#12-virtual-gamepad-permissions)
13. [Camera check](#13-camera-check)
14. [Verify the install](#14-verify-the-install)
15. [Run in simulation](#15-run-in-simulation)
16. [Run on the real G1](#16-run-on-the-real-g1)
17. [Re-recording gestures](#17-re-recording-gestures)
18. [Troubleshooting](#18-troubleshooting)
19. [Tested versions](#19-tested-versions)

---

## 1. What you need

| Item | Requirement |
| --- | --- |
| OS | **Ubuntu 22.04 LTS**, native install recommended. A VirtualBox VM works (see note below). WSL2 and Docker are not supported: CycloneDDS needs direct access to network interfaces, and the virtual gamepad needs `/dev/uinput`. |
| Desktop session | **Ubuntu on Xorg** (choose it from the cog icon on the login screen). The program sends key presses to the simulator window with `xdotool`, which does not work under Wayland. |
| Python | 3.10 (Ubuntu 22.04's default). |
| CPU / RAM / disk | 4+ cores, 8 GB RAM, about 15 GB free disk. |
| GPU | Not required. Without one, the simulator renders in software and uses more CPU. |
| Webcam | Any USB or built-in webcam. |
| Internet | Needed during setup to clone repositories and download packages and models. |
| Repo access | The project repository is private. You need a GitHub account with access to `A5troGaze/CITS3200_Project`, or a copy of the code from the team. |

**Running in a VirtualBox VM?** With the VM powered off:
- Settings > System > Processor: no more virtual CPUs than the host has physical cores (for example 4 to 6). Over-allocating makes everything stall.
- Settings > Display: enable 3D acceleration, 128 MB video memory.
- Install the VirtualBox Extension Pack on the host (required for webcam passthrough).
- With the VM running: Devices > Webcams > tick your camera.

---

## 2. How the system fits together

Three programs run at once, each in its own terminal. They talk over DDS (the same messaging layer the real robot uses) on the local loopback interface `lo`, DDS domain `0`.

```
 Webcam
   |
   v
 run.py  (this repo)
   |  Gesture mode: hand gesture -> virtual Xbox gamepad stick movements
   |  Mimic mode:   body pose -> arm joint angles on DDS topic rt/arm_sdk
   v
 unitree_mujoco  (simulator, reads the gamepad, publishes robot state)
   ^
   |  DDS: rt/lowstate, rt/lowcmd
   v
 g1_ctrl  (trained walking policy from unitree_rl_lab: keeps the robot balanced and walking)
```

- **Gesture mode:** `run.py` recognises one of 8 hand gestures and moves the sticks of a virtual Xbox 360 controller. The simulator passes the stick state to `g1_ctrl`, which walks the robot.
- **Mimic mode:** `run.py` tracks the operator's body, retargets the arm pose onto the G1 with GMR (General Motion Retargeting), and publishes arm joint targets on `rt/arm_sdk`. The patched simulator blends these into the arm motors while `g1_ctrl` keeps the legs balanced.
- **Real robot:** `run.py` talks to the robot's built-in locomotion service directly. The simulator and `g1_ctrl` are not used. Only gesture mode is supported on the real robot.

---

## 3. Folder layout

The program finds its third-party files by looking for a folder called `Dependencies` **next to** the project folder. The parent folder can be called anything; this guide uses `~/HumanoidControl`.

```
~/
├── cyclonedds/
│   └── install/                        <- CycloneDDS 0.10.2 (step 6)
├── .mujoco/
│   └── mujoco-3.3.6/                   <- MuJoCo C++ release (step 10)
└── HumanoidControl/
    ├── Project/                        <- this git repository
    │   ├── run.py                      <- the program you run
    │   ├── src/                        <- program modules and data (gestures.json, IK config)
    │   ├── gesture_library/            <- photos of each gesture
    │   └── unitree_mujoco_patch/       <- modified simulator file (step 10)
    └── Dependencies/                   <- must be named exactly this
        ├── g1-env/                     <- Python virtual environment (step 7)
        ├── Models/
        │   ├── hand_landmarker.task    <- MediaPipe hand model (step 8)
        │   └── pose_landmarker.task    <- MediaPipe pose model (step 8)
        ├── unitree_sdk2_python/        <- Python DDS SDK (step 7)
        ├── GMR/                        <- motion retargeting library (step 7)
        ├── unitree_sdk2/               <- C++ DDS SDK (step 9)
        ├── unitree_mujoco/             <- simulator (step 10)
        └── unitree_rl_lab/             <- walking controller (step 11)
```

The folders `legacy_hand_gestures/` and `legacy_pose_mimic/` in the repo hold earlier standalone versions of each subsystem and their development notes. They are not needed to run the program.

---

## 4. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential cmake git curl wget \
  python3 python3-pip python3-venv python3-dev \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev libglfw3-dev \
  libgl1 libegl1 libglib2.0-0 libevdev2 \
  xdotool v4l-utils
```

What these are for:
- `build-essential`, `cmake`, the `lib*-dev` packages: compiling the Unitree C++ SDK, the simulator and `g1_ctrl`.
- `libgl1`, `libegl1`, `libglfw3-dev`: the simulator window.
- `libevdev2`: used by the virtual gamepad library.
- `xdotool`: lets the program press keys in the simulator window (lowering and releasing the robot's support band).
- `v4l-utils`: webcam diagnostics.

---

## 5. Get the project code

The repository is private, so git needs your GitHub login. The simplest way is the GitHub CLI:

```bash
sudo apt install -y gh
gh auth login      # GitHub.com > HTTPS > Yes (authenticate Git) > Login with a web browser
gh auth status     # should say "Logged in to github.com"
```

Then clone:

```bash
mkdir -p ~/HumanoidControl/Dependencies/Models
cd ~/HumanoidControl
git clone https://github.com/A5troGaze/CITS3200_Project.git Project
cd Project
git checkout Cleanup
ls run.py src    # both must exist
```

The release code is on the `Cleanup` branch. Once it has been merged, use `git checkout main` instead. If `run.py` or `src/` is missing, you are on an older branch.

---

## 6. CycloneDDS 0.10.2

The DDS messaging library that the Unitree Python SDK is built on. It is built from source at a pinned version.

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && git checkout 0.10.2
mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install            # a few minutes

echo 'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"' >> ~/.bashrc
source ~/.bashrc
echo $CYCLONEDDS_HOME                        # should print /home/<you>/cyclonedds/install
```

---

## 7. Python environment

### 7.1 Create the virtual environment

```bash
python3 -m venv ~/HumanoidControl/Dependencies/g1-env
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
pip install --upgrade pip
```

Activate this environment in **every** new terminal before running any project command:

```bash
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
```

A virtual environment records its own absolute path. If you move or rename `Dependencies/`, delete `g1-env/` and repeat this step.

### 7.2 Clone the Python libraries

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/YanjieZe/GMR.git
git -C unitree_sdk2_python checkout 65691c8
git -C GMR checkout bb1bbe4
```

The `checkout` lines pin the versions the project was tested with. Newer versions will probably work; if something breaks, come back to these.

### 7.3 Install the packages

```bash
# CPU-only PyTorch first. GMR depends on PyTorch, and the default download
# pulls several GB of GPU libraries that are not used.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Unitree Python SDK (needs CYCLONEDDS_HOME from step 6)
pip install -e ~/HumanoidControl/Dependencies/unitree_sdk2_python

# GMR (also installs mink, daqp, scipy, opencv-python and others)
pip install -e ~/HumanoidControl/Dependencies/GMR

# Everything else
pip install mujoco==3.11.0 mediapipe==1.0.0 vgamepad pyyaml
```

`-e` installs the library from its folder in place, so the folder must stay where it is.

---

## 8. MediaPipe models

Two model files, kept outside the repo:

```bash
cd ~/HumanoidControl/Dependencies/Models

# Hand model (gesture mode)
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Pose model (mimic mode)
curl -L -o pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

ls -l    # both files should be several MB; pose_landmarker.task is about 5.8 MB
```

The file names must be exactly `hand_landmarker.task` and `pose_landmarker.task`.

---

## 9. Unitree C++ SDK (unitree_sdk2)

Both the simulator and `g1_ctrl` are compiled against this. It must be installed to `/usr/local` (the default), because `g1_ctrl` looks for it there.

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install
sudo ldconfig
```

Check:

```bash
ls /usr/local/lib/libddsc.so* /usr/local/lib/libddscxx.so*    # both must list files
```

---

## 10. MuJoCo simulator (unitree_mujoco, patched)

The simulator is Unitree's C++ build of MuJoCo with DDS support. The project ships one modified simulator file, `unitree_mujoco_patch/unitree_sdk2_bridge.h`, which adds the `rt/arm_sdk` topic so mimic mode can control the arms while `g1_ctrl` balances the legs. The patch is written against one specific simulator commit, so the checkout below is required, not optional.

### 10.1 MuJoCo C++ library

The simulator needs the MuJoCo 3.3.6 C++ release. This is separate from the `mujoco` Python package installed in step 7.

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz
tar xzf mujoco-3.3.6-linux-x86_64.tar.gz
ls ~/.mujoco/mujoco-3.3.6    # should contain bin include lib simulate ...
```

### 10.2 Clone, patch and build

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_mujoco.git
cd unitree_mujoco
git checkout 4134cb5dc7ff1ba7f484deda48b5274b58694519

cd simulate
ln -s ~/.mujoco/mujoco-3.3.6 mujoco

# Install the project's patched bridge (the sed strips the notes block at the top of the file)
sed "/^'''$/,/^'''$/d" ~/HumanoidControl/Project/unitree_mujoco_patch/unitree_sdk2_bridge.h \
  > src/unitree_sdk2_bridge.h

mkdir build && cd build
cmake .. && make -j4
```

This produces `simulate/build/unitree_mujoco`.

### 10.3 Configure

Edit `~/HumanoidControl/Dependencies/unitree_mujoco/simulate/config.yaml` so these lines read:

```yaml
robot: "g1"
robot_scene: "scene.xml"
domain_id: 0
interface: "lo"
use_joystick: 1
joystick_type: "xbox"
joystick_device: "/dev/input/js0"
enable_elastic_band: 1
```

Leave the other lines as they are. Why each matters:
- `domain_id: 0`: `g1_ctrl` is hard-coded to DDS domain 0. If the simulator uses any other domain, the two start without errors but never see each other.
- `use_joystick: 1`: the simulator reads the virtual gamepad created by `run.py`.
- `enable_elastic_band: 1`: a virtual support band holds the robot up while it gets into a standing pose. The program lowers and releases it automatically.
- `joystick_device`: the virtual gamepad normally becomes `/dev/input/js0`. If a physical controller is also plugged in, it may become `js1` instead (see Troubleshooting).

### 10.4 Quick test

```bash
cd ~/HumanoidControl/Dependencies/unitree_mujoco/simulate/build
./unitree_mujoco
```

A window titled MuJoCo should open with the G1 hanging from the band. Close it. (It may print a joystick warning at this point because no gamepad exists yet; that is expected.)

---

## 11. Walking controller (g1_ctrl)

`g1_ctrl` runs Unitree's pretrained walking policy (an ONNX neural network). No training and no GPU are needed.

```bash
cd ~/HumanoidControl/Dependencies
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git -C unitree_rl_lab checkout 4960b84

# ONNX Runtime (extracts into the existing thirdparty folder)
cd ~/HumanoidControl/Dependencies/unitree_rl_lab/deploy/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz

# Build the G1 29-DOF controller
cd ~/HumanoidControl/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make
```

This produces `g1_29dof/build/g1_ctrl`.

---

## 12. Virtual gamepad permissions

Gesture mode creates a virtual Xbox 360 controller through `/dev/uinput`, which normal users cannot access by default. Grant access permanently:

```bash
echo 'KERNEL=="uinput", TAG+="uaccess", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/50-uinput.rules
sudo usermod -aG input $USER
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**Log out and back in** (or reboot) so the group change applies. Then check:

```bash
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
python3 -c "import vgamepad as vg; vg.VX360Gamepad(); print('gamepad OK')"
```

If this fails with a permission error, run `groups` and confirm `input` is listed. As a temporary workaround for one session: `sudo chmod 0666 /dev/uinput`.

---

## 13. Camera check

```bash
ls /dev/video*          # at least one device, e.g. /dev/video0
v4l2-ctl --list-devices
groups $USER            # should include "video"
```

If `video` is missing from your groups: `sudo usermod -aG video $USER`, then log out and back in.

The program uses camera index 0 (the first camera). If the machine has more than one camera and the wrong one opens, unplug the others or disable the built-in one.

**Insta360 and some other USB cameras** can give corrupted colours in OpenCV on first connection. Open the Cheese webcam app once (`sudo apt install cheese`, run `cheese`, confirm a picture, close it) at the start of each session before starting the program.

---

## 14. Verify the install

```bash
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
python3 -c "import unitree_sdk2py, mediapipe, mujoco, cv2, vgamepad, general_motion_retargeting; print('all imports OK')"
```

A line saying `xrobotoolkit_sdk not found, skip for now` is normal (an optional GMR feature).

Then test gesture recognition on its own, with no simulator:

```bash
cd ~/HumanoidControl/Project
python3 src/recognize_hand_gestures.py
```

A camera window opens with a skeleton drawn over your hand. Make the gestures shown in `gesture_library/`; the recognised gesture name prints in the terminal when it changes. Press `q` to quit.

---

## 15. Run in simulation

You need three terminals. Activate the environment in each one first:

```bash
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
```

Start them **in this order**. `run.py` must be first, because it creates the virtual gamepad and the simulator only looks for a gamepad when it starts.

**Terminal 1: the program**

```bash
cd ~/HumanoidControl/Project
python3 run.py sim
```

It prints `Virtual gamepad is live` and waits. Leave it waiting.

**Terminal 2: the simulator**

```bash
cd ~/HumanoidControl/Dependencies/unitree_mujoco/simulate/build
./unitree_mujoco
```

**Terminal 3: the walking controller**

```bash
cd ~/HumanoidControl/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

Wait for `Connected to robot.` If it does not appear, see Troubleshooting. `--network lo` is required: without it `g1_ctrl` may pick a different network interface from the simulator and never connect.

**Back in Terminal 1: start-up sequence**

1. Press **Enter**. The robot moves into its standing pose (the program holds L2 + D-pad Up on the virtual gamepad).
2. The program loosens the support band in steps and asks `Are the robot's feet on the ground?`. Watch the MuJoCo window. Press **Enter** to loosen more, or type **y** and press Enter once both feet are flat on the floor. Do not move or cover the MuJoCo window during this step; key presses are sent to it.
3. The program then starts the walking policy (R1 + X) and releases the band automatically.
4. A camera window titled **Humanoid Control** opens.

**Using it**

In the camera window:

| Key | Action |
| --- | --- |
| `m` | Switch between gesture mode and mimic mode |
| `q` | Quit (the robot stops and the arms are handed back cleanly) |

**Gesture mode** (starts in this mode). Hold a gesture to move; drop your hand to stop. Photos of each pose are in `gesture_library/`.

| Gesture | Robot action (simulation) |
| --- | --- |
| `move_forward_left_hand` / `move_forward_right_hand` | Walk forward |
| `move_backward_left_hand` / `move_backward_right_hand` | Walk backward |
| `move_left` | Step sideways left |
| `move_right` | Step sideways right |
| `turn_left` | Turn left on the spot (alternates small forward and backward steps while turning) |
| `turn_right` | Turn right on the spot (alternates small forward and backward steps while turning) |

The status line at the top of the camera window shows the recognised gesture, or `no match`.

**Mimic mode.** Stand back so your upper body is fully in frame. The robot stops walking and its arms follow yours. The first switch into mimic mode takes a few seconds to load the pose and retargeting models, and the camera image freezes during that time. Press `m` again to return the arms to the walking controller and go back to gesture mode. Move your arms at a moderate speed; arm speed is capped at 1 rad/s because faster two-arm movements can tip the robot over.

**Gesture sensitivity.** The optional second argument sets the match threshold (default 1.5). Higher is looser, lower is stricter:

```bash
python3 run.py sim 2.0
```

**Shutting down.** Press `q` in the camera window, then `Ctrl+C` in Terminals 3 and 2 (or close the MuJoCo window).

---

## 16. Run on the real G1

> **Safety.** Test everything in simulation first. Have someone holding the Unitree remote with the emergency stop ready, keep a clear area of at least 3 m around the robot, and use a gantry or harness for the first runs. Only gesture mode works on the real robot; mimic mode is not implemented for hardware and does nothing.

The simulator and `g1_ctrl` are **not** used. `run.py` sends walking commands straight to the robot's built-in locomotion service.

1. **Connect to the robot's network.** Plug the computer into the G1 with an Ethernet cable, then give the computer's wired connection a static address in the robot's subnet, for example IP `192.168.123.99`, netmask `255.255.255.0` (Settings > Network > Wired > IPv4 > Manual). Check the link with `ping 192.168.123.164` (the G1's onboard computer).
2. **Find the interface name.** Run `ip -br addr` and note the interface holding `192.168.123.99`, for example `enp3s0` or `eth0`.
3. **Set the interface in the code.** The real-robot controller defaults to `wlan0`. Open `src/real_controller.py` and change the default in `def __init__(self, interface="wlan0"):` to your interface name.
4. **Prepare the robot.** Power on the G1 and, using the handheld remote, bring it into its normal standing locomotion mode as described in Unitree's G1 user manual. The robot must already be standing and able to walk from the remote.
5. **Run:**

   ```bash
   source ~/HumanoidControl/Dependencies/g1-env/bin/activate
   cd ~/HumanoidControl/Project
   python3 run.py real
   ```

Real-robot speeds are deliberately lower than in simulation (0.3 m/s walking, 0.3 rad/s turning). Dropping your hand or losing the gesture sends a stop command. Pressing `q` sends a final stop before exiting.

---

## 17. Re-recording gestures

The recognition data in `src/data/gestures.json` was recorded by the team (20 samples per gesture). If recognition is unreliable for a new operator, camera or lighting, record new samples:

```bash
cd ~/HumanoidControl/Project
python3 src/record_hand_gestures.py
```

Hold a gesture and press its number key; 20 samples are captured. Press `s` to save and quit.

| Key | Gesture |
| --- | --- |
| 1 | `turn_right` |
| 2 | `turn_left` |
| 3 | `move_right` |
| 4 | `move_left` |
| 5 | `move_forward_left_hand` |
| 6 | `move_forward_right_hand` |
| 7 | `move_backward_left_hand` |
| 8 | `move_backward_right_hand` |

New samples are **added** to the existing file. To start from scratch, back up and remove `src/data/gestures.json` first, then record all 8 gestures before pressing `s`. Check the result with `python3 src/recognize_hand_gestures.py`.

---

## 18. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `pip install -e unitree_sdk2_python` fails with `Could not locate cyclonedds` | `CYCLONEDDS_HOME` is not set in this terminal. Run `source ~/.bashrc`, check `echo $CYCLONEDDS_HOME`, retry. |
| `FileNotFoundError: Couldn't find .../hand_landmarker.task` | The `Dependencies` folder is not next to the project folder, or the model is missing or misnamed. See steps 3 and 8. |
| `unitree_mujoco` fails to compile with an error near `'''` in `unitree_sdk2_bridge.h` | The notes block at the top of the patched file was copied in. Redo the `sed` command in step 10.2. |
| `unitree_mujoco` build cannot find `mujoco/mujoco.h` | The `mujoco` symlink in `simulate/` is missing or points to the wrong folder. `ls -l simulate/mujoco` should point to `~/.mujoco/mujoco-3.3.6`. |
| A program fails with `error while loading shared libraries` after building | Run `sudo ldconfig`. |
| `g1_ctrl` never prints `Connected to robot.` | `domain_id` in the simulator's `config.yaml` is not `0`, `interface` is not `"lo"`, or `g1_ctrl` was started without `--network lo`. |
| Robot does not respond to the start-up sequence or gestures | The simulator did not find the gamepad. Start `run.py` first, then the simulator. Run `ls /dev/input/js*`: if the virtual pad is `js1`, set `joystick_device: "/dev/input/js1"` in `config.yaml`. |
| Support band does not lower or release | `xdotool` cannot reach the window. Log in with **Ubuntu on Xorg**, not Wayland (`echo $XDG_SESSION_TYPE` should print `x11`). Keep the MuJoCo window open and uncovered. As a fallback, click the MuJoCo window and press `8` to lower, `9` to release. |
| `PermissionError` on `/dev/uinput` | Step 12 not done, or you have not logged out since. |
| `Could not open camera` or a black camera window | Check `ls /dev/video*`. In VirtualBox: Devices > Webcams. Close any other app using the camera. |
| Colours look wrong in the camera window | Open and close Cheese once first (step 13). |
| Mimic mode prints `Couldn't start MimicSource` | Mimic mode could not load, and the program keeps running in gesture mode. Read the error after the colon. Usually GMR is not installed (step 7) or `pose_landmarker.task` is missing (step 8). |
| Arms do not move in mimic mode in the simulator | The simulator was built without the project's patched bridge. Redo step 10.2 and rebuild. |
| Robot falls over | Restart all three programs. Move your arms more slowly in mimic mode. In the MuJoCo window, `9` re-attaches the band. |
| Simulator runs in slow motion | Not enough CPU. Close other programs; in a VM, check the CPU settings in section 1. |
| Real robot does not move | Wrong interface name in `src/real_controller.py`, computer not on `192.168.123.x`, or the robot is not in standing locomotion mode. |

---

## 19. Tested versions

| Component | Version |
| --- | --- |
| Ubuntu | 22.04.5 |
| Python | 3.10.12 |
| CycloneDDS | 0.10.2 |
| unitree_sdk2_python | commit `65691c8` |
| GMR | commit `bb1bbe4` |
| unitree_mujoco | commit `4134cb5` (required by the patch) |
| MuJoCo C++ (simulator) | 3.3.6 |
| unitree_rl_lab | commit `4960b84` |
| ONNX Runtime | 1.22.0 |
| mujoco (Python) | 3.11.0 |
| mediapipe | 1.0.0 |
| opencv-python | 5.0.0.93 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| torch | 2.13.0+cpu |
