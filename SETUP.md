# Setup Guide

This guide installs everything needed to run this G1 control software on a fresh Ubuntu 22.04 machine. When finished: see [USAGE.md](USAGE.md) for how to run the program and register your own custom gestures.

Follow steps in order; the latter are dependent of the earlier.

---

## Contents

1. [Requirements](#1-requirements)
2. [Folder layout](#2-folder-layout)
3. [Get the project code](#3-get-the-project-code)
4. [System packages](#4-system-packages)
5. [CycloneDDS](#5-cyclonedds)
6. [Third-party repositories](#6-third-party-repositories)
7. [Python environment](#7-python-environment)
8. [MediaPipe models](#8-mediapipe-models)
9. [Unitree C++ SDK](#9-unitree-c-sdk)
10. [MuJoCo](#10-mujoco)
11. [Simulator (unitree_mujoco)](#11-simulator-unitree_mujoco)
12. [Walking controller (g1_ctrl)](#12-walking-controller-g1_ctrl)
13. [Virtual gamepad access](#13-virtual-gamepad-access)
14. [Verify the install](#14-verify-the-install)
15. [Troubleshooting](#15-troubleshooting)
16. [Tested versions](#16-tested-versions)

---

## 1. Requirements

| | |
| --- | --- |
| **OS** | Ubuntu 22.04 LTS |
| **Python** | 3.10 (Ubuntu 22.04's default) |
| **Hardware** | 4+ CPU cores, 8 GB RAM, 15 GB free disk. A GPU is not required. |
| **Webcam** | Any USB or built-in webcam |
| **Internet** | Needed during setup |

**Display session.** The program presses keys in the simulator window using `xdotool`, which only works under Xorg, not Wayland. Check which one you are using:

```bash
echo $XDG_SESSION_TYPE
```

If it prints `x11`, you are ready. If it prints `wayland`, log out, click your username on the login screen, click the gear icon in the bottom-right corner, choose **Ubuntu on Xorg**, and log in again. Ubuntu remembers this choice.

---


## 2. Folder layout

The program finds its third-party files by looking for a folder named `Dependencies/` **next to** the project folder. Both sit inside one parent folder.

The parent folder and the project folder can have any name and live anywhere. For simplicity, this guide uses `~/HumanoidControl/` and `G1Control/`. If you choose different names, adjust the paths in each command.

```text
~/
├── cyclonedds/                      CycloneDDS, built from source (step 5)
└── HumanoidControl/
    ├── G1Control/                   this repository
    └── Dependencies/                must be named exactly this
        ├── g1-env/                  Python virtual environment (step 7)
        ├── Models/                  MediaPipe models (step 8)
        ├── GMR/                     motion retargeting library (step 6)
        ├── unitree_sdk2_python/     Unitree Python SDK (step 6)
        ├── unitree_sdk2/            Unitree C++ SDK (step 6)
        ├── unitree_mujoco/          simulator (step 6)
        ├── unitree_rl_lab/          walking controller (step 6)
        └── mujoco-3.3.6/            MuJoCo library used by the simulator (step 10)
```

---


## 3. Get the project code

Install `git` and `curl`, which the rest of the guide relies on:

```bash
sudo apt update
sudo apt install -y git curl
```

Create the folders and clone the repository:

```bash
mkdir -p ~/HumanoidControl/Dependencies/Models
cd ~/HumanoidControl
git clone https://github.com/A5troGaze/CITS3200_Project.git G1Control
```

---


## 4. System packages

The list of required system packages is kept in `apt-packages.txt` in the repository. Install them all at once:

```bash
cd ~/HumanoidControl/G1Control
xargs -a apt-packages.txt sudo apt install -y
```

---


## 5. CycloneDDS

CycloneDDS is the messaging library the Unitree SDK communicates over. It is built from source at version 0.10.2.

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
git checkout 0.10.2
mkdir build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

Tell other software where CycloneDDS is installed, and load the setting into this terminal:

```bash
echo 'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"' >> ~/.bashrc
source ~/.bashrc
```

Check that it is set:

```bash
echo $CYCLONEDDS_HOME
```

This should print `/home/<your-username>/cyclonedds/install`.

---


## 6. Third-party repositories

Clone the Unitree and GMR repositories into `Dependencies/`, then pin each one to the exact version the project was tested with:

```bash
cd ~/HumanoidControl/Dependencies

git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/YanjieZe/GMR.git
git clone https://github.com/unitreerobotics/unitree_sdk2.git
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/unitreerobotics/unitree_rl_lab.git

git -C unitree_sdk2_python checkout 65691c8
git -C GMR                 checkout bb1bbe4
git -C unitree_mujoco      checkout 4134cb5
git -C unitree_rl_lab      checkout 4960b84
```

The `unitree_mujoco` version is required: the project's simulator patch (step 11) is written for that exact version.

---


## 7. Python environment

Create and activate the virtual environment:

```bash
python3 -m venv ~/HumanoidControl/Dependencies/g1-env
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
pip install --upgrade pip
```

Install all Python packages from `requirements.txt`. Run this from the project folder, because the file refers to `../Dependencies/`:

```bash
cd ~/HumanoidControl/G1Control
pip install -r requirements.txt
```

> **Note:** If this fails with `Could not locate cyclonedds`, `CYCLONEDDS_HOME` is not set in this terminal. Run `source ~/.bashrc` and try again.

> **Note:** A virtual environment stores its own location. If you later move or rename the `Dependencies` folder, delete `g1-env/` and repeat this step.

---


## 8. MediaPipe models

Download the hand model (gesture mode) and the pose model (mimic mode). The file names must be exactly `hand_landmarker.task` and `pose_landmarker.task`.

```bash
cd ~/HumanoidControl/Dependencies/Models

curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

curl -L -o pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Check the downloads:

```bash
ls -l
```

`hand_landmarker.task` should be about 7.8 MB and `pose_landmarker.task` about 5.8 MB.

---


## 9. Unitree C++ SDK

The simulator and the walking controller are both compiled against this SDK. It installs to `/usr/local`, where the walking controller expects to find it.

```bash
cd ~/HumanoidControl/Dependencies/unitree_sdk2
mkdir build
cd build
cmake .. -DBUILD_EXAMPLES=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

---


## 10. MuJoCo

The simulator is built on the MuJoCo 3.3.6 C++ library. (This is separate from the `mujoco` Python package installed in step 7.)

```bash
cd ~/HumanoidControl/Dependencies
curl -L -O https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz
tar -xzf mujoco-3.3.6-linux-x86_64.tar.gz
rm mujoco-3.3.6-linux-x86_64.tar.gz
```

This creates `Dependencies/mujoco-3.3.6/`.

---


## 11. Simulator (unitree_mujoco)

### 11.1 Link MuJoCo and install the project's files

The simulator's build expects MuJoCo at `simulate/mujoco`. Create a link there that points to the folder from step 10:

```bash
cd ~/HumanoidControl/Dependencies/unitree_mujoco/simulate
ln -s ../../mujoco-3.3.6 mujoco
```

Copy in the project's two simulator files from `unitree_mujoco_patch/`:

```bash
cp ~/HumanoidControl/G1Control/unitree_mujoco_patch/unitree_sdk2_bridge.h src/unitree_sdk2_bridge.h
cp ~/HumanoidControl/G1Control/unitree_mujoco_patch/config.yaml config.yaml
```

- `unitree_sdk2_bridge.h` replaces the simulator's DDS bridge with a patched version that lets mimic mode control the robot's arms.
- `config.yaml` replaces the simulator's settings. It loads the 29-joint G1 (`scene_29dof.xml`), uses DDS domain `0` on the loopback interface `lo` (matching `g1_ctrl`), reads the virtual Xbox gamepad from `/dev/input/js0`, and enables the support band used during start-up.

### 11.2 Build

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

This produces `simulate/build/unitree_mujoco`.

### 11.3 Check the build

```bash
ls -l ~/HumanoidControl/Dependencies/unitree_mujoco/simulate/build/unitree_mujoco
```

The file should be listed. Do not start the simulator on its own: the project's `config.yaml` tells it to read the virtual gamepad, which only exists while `run.py` is running. Started alone, it prints `Error: Joystick open failed.` and closes (sometimes with a `Segmentation fault`). It is started properly in [USAGE.md](USAGE.md#3-run-in-simulation).

---


## 12. Walking controller (g1_ctrl)

`g1_ctrl` runs Unitree's pretrained walking policy. No training or GPU is needed, and the ONNX Runtime library it uses is already included in the repository.

```bash
cd ~/HumanoidControl/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof
mkdir build
cd build
cmake ..
make -j$(nproc)
```

This produces `g1_29dof/build/g1_ctrl`.

---

## 13. Virtual gamepad access

The program creates a virtual Xbox controller through `/dev/uinput`, which normal users cannot access by default. Add a rule that grants access to the logged-in user:

```bash
echo 'KERNEL=="uinput", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/50-uinput.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Log out and log back in** for the rule to take effect.

---


## 14. Verify the install

Open a new terminal and activate the environment:

```bash
source ~/HumanoidControl/Dependencies/g1-env/bin/activate
```

Check that every Python library loads:

```bash
python -c "import unitree_sdk2py, general_motion_retargeting, mediapipe, mujoco, cv2, scipy, vgamepad; print('imports OK')"
```

A line saying `XRobotStreamer is not installed` is normal; it is an optional GMR feature that is not used.

Check that the virtual gamepad can be created:

```bash
python -c "import vgamepad; vgamepad.VX360Gamepad(); print('gamepad OK')"
```

Check that a camera is detected:

```bash
ls /dev/video*
```

At least one device, such as `/dev/video0`, should be listed.

Setup is complete. Continue with [USAGE.md](USAGE.md).


## 15. Troubleshooting

| Problem | Fix |
| --- | --- |
| `Could not locate cyclonedds` during `pip install` | `CYCLONEDDS_HOME` is not set in this terminal. Run `source ~/.bashrc`, check it with `echo $CYCLONEDDS_HOME`, and retry. |
| Simulator build cannot find `mujoco/mujoco.h` | The `mujoco` link in `simulate/` is missing or wrong. `ls -l ~/HumanoidControl/Dependencies/unitree_mujoco/simulate/mujoco` should point to `../../mujoco-3.3.6`. |
| Simulator opens a different robot, or no robot | The project's `config.yaml` was not copied. Redo the second `cp` command in step 11.1. |
| `error while loading shared libraries` when starting a program | Run `sudo ldconfig`. |
| `PermissionError` on `/dev/uinput` | Step 13 was not done, or you have not logged out and back in since. |
| `ls /dev/video*` shows nothing | The camera is not detected. Reconnect it, or try a different USB port. |

---


## 16. Tested versions

| Component | Version |
| --- | --- |
| Ubuntu | 22.04.5 |
| Python | 3.10.12 |
| CycloneDDS | 0.10.2 |
| unitree_sdk2_python | `65691c8` |
| GMR | `bb1bbe4` |
| unitree_mujoco | `4134cb5` |
| unitree_rl_lab | `4960b84` |
| MuJoCo (C++ library) | 3.3.6 |
| ONNX Runtime | 1.22.0 (bundled with unitree_rl_lab) |
| Python packages | see `requirements.txt` |