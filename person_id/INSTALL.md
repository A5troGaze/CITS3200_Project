# person_id: install and test on a new machine

Everything needed to set up the person_id program from scratch and check it
works: a simulated Unitree G1 that copies the arm movements of a person
in front of a webcam, balanced by unitree_rl_lab's walking policy.

Day-to-day commands and troubleshooting are in [RUNNING.md](RUNNING.md);
this file only gets a new machine to the point where they work.

---

## 0. What you need

| | |
| --- | --- |
| OS | **Ubuntu 22.04** (tested on 22.04.5). Other Linux distributions may work but are untested. Windows/macOS: use an Ubuntu VM or WSL2 (the webcam is easier in a VM). |
| Python | **3.10** (Ubuntu 22.04's default). GMR needs 3.10 or newer. |
| CPU / RAM / disk | 4+ cores, 8 GB RAM, ~10 GB free (PyTorch, pulled in by GMR, is the biggest download) |
| GPU | Not required. Without one the simulator window is rendered in software and uses a lot of CPU. |
| Webcam | Any USB webcam, only needed for the live demo (tests and replays don't need one) |
| Internet | To clone repositories and download Python packages and the pose model |

**Running in a VirtualBox VM?** Do this first, with the VM powered off:
- Settings -> System -> Processor: CPU count **no more than the host's
  physical cores** (e.g. 4-6). Too many virtual CPUs makes everything stall.
- Settings -> Display: enable 3D acceleration, 128 MB video memory.
- Install the VirtualBox Extension Pack on the host (needed for the webcam).
- With the VM running: Devices -> Webcams -> tick your camera.

## 1. Folder layout

Everything lives under `~/CITS3200`; third-party code stays outside the git repo:

```
~/CITS3200/
├── Project/                  <- this git repository
└── Dependencies/
    ├── g1-env/               <- Python virtual environment
    ├── unitree_sdk2_python/  <- Unitree SDK (DDS messages)
    ├── unitree_mujoco/       <- G1 simulator
    ├── GMR/                  <- motion retargeting library
    ├── unitree_rl_lab/       <- balance (walking) policy
    └── Models/               <- MediaPipe pose model
~/cyclonedds/                 <- CycloneDDS C library (built from source)
```

## 2. System packages

```bash
sudo apt update
sudo apt install -y git curl python3 python3-pip python3-venv \
    cmake build-essential v4l-utils libgl1 libegl1 libglib2.0-0
```

## 3. CycloneDDS 0.10.2 (needed by the Unitree SDK)

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && git checkout 0.10.2
mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install            # takes a few minutes
echo 'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"' >> ~/.bashrc
source ~/.bashrc
echo $CYCLONEDDS_HOME                         # should print /home/<you>/cyclonedds/install
```

## 4. This repository

```bash
mkdir -p ~/CITS3200/Dependencies/Models
cd ~/CITS3200
git clone https://github.com/A5troGaze/CITS3200_Project.git Project   # private: you need access
cd Project
git checkout person_id_finalised
```

## 5. Third-party repositories

```bash
cd ~/CITS3200/Dependencies
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/YanjieZe/GMR.git
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
```

These are the commits person_id was tested with. Newer ones probably
work; if something breaks, check these out
(`git -C <folder> checkout <commit>`):

| Repository | Tested commit |
| --- | --- |
| unitree_sdk2_python | `65691c8` |
| unitree_mujoco | `1eb6642` |
| GMR | `bb1bbe4` |
| unitree_rl_lab | `4960b84` |

Only unitree_rl_lab's pretrained policy files are used
(`deploy/robots/g1_29dof/config/policy/velocity/v0/`): no IsaacLab, no
training, no GPU.

## 6. Python environment and packages

```bash
cd ~/CITS3200/Dependencies
python3 -m venv g1-env
source ~/CITS3200/Dependencies/g1-env/bin/activate      # do this in every new terminal
pip install --upgrade pip

# CPU-only PyTorch first (GMR depends on it; the default wheel is several GB of GPU libraries)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Unitree SDK (needs CYCLONEDDS_HOME from step 3)
pip install -e ~/CITS3200/Dependencies/unitree_sdk2_python

# GMR (also installs mink, qpsolvers/daqp, scipy, opencv-python, smplx, ...)
pip install -e ~/CITS3200/Dependencies/GMR

# The rest
pip install mujoco==3.11.0 mediapipe==1.0.0 onnxruntime pyyaml pygame pytest
```

Versions person_id was tested with: Python 3.10.12, mujoco 3.11.0,
mediapipe 1.0.0, numpy 2.2.6, scipy 1.15.3, mink 1.2.0, daqp 0.8.7,
qpsolvers 4.13.0, onnxruntime 1.23.2, opencv-python 5.0.0.93,
cyclonedds 0.10.2, PyYAML 6.0.3, pygame 2.6.1, pytest 9.1.1, torch 2.13.0+cpu.

Check the imports:
```bash
python3 -c "import unitree_sdk2py, mediapipe, mujoco, general_motion_retargeting, onnxruntime; print('all OK')"
```
A line saying `xrobotoolkit_sdk not found, skip for now` is normal (an optional GMR feature).

## 7. Pose model

```bash
curl -L -o ~/CITS3200/Dependencies/Models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
ls -l ~/CITS3200/Dependencies/Models/pose_landmarker.task    # about 5.8 MB
```

## 8. Simulator settings

Open `~/CITS3200/Dependencies/unitree_mujoco/simulate_python/config.py` and
make sure it says:

```python
ROBOT = "g1"
DOMAIN_ID = 1
INTERFACE = "lo"
```

(Leave the rest alone. person_id starts the simulator through its own
launcher, `person_id/sim_standing.py`, which reads this file.)

## 9. Test it (no webcam or simulator needed)

```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
cd ~/CITS3200/Project
python3 -m pytest person_id/tests -q
```
Expect `130 passed` in under a minute. This checks the retargeting, the
arm-copying accuracy on 26 test poses, the joint mapping against the
simulator's robot model, the balance policy wiring and the publishers.

## 10. Run it

All from `~/CITS3200/Project/person_id`, with the environment activated:
```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
cd ~/CITS3200/Project/person_id
mkdir -p exports          # scratch folder for recordings and videos (gitignored)
```

**a. Without a webcam or simulator (quick check):**
```bash
python3 make_synthetic_replay.py exports/synthetic.json
python3 person_id_replay.py exports/synthetic.json --dry-run --record-demo exports/synthetic_demo.mp4
```
Writes a video of a stick figure next to the G1 copying it.

**b. With the simulator (no webcam):**
```bash
# terminal 1
python3 sim_standing.py --rl-lab
# terminal 2, once terminal 1 says "Band released"
python3 make_synthetic_replay.py exports/gentle.json --poses arms_down,wave,t_pose,goalpost,elbows_90_forearms_forward
python3 person_id_replay.py exports/gentle.json --report-tracking
```
The G1 stands up, the elastic band fades out after ~8 s, and it then
balances on its own while its arms follow the replay. The simulator should
print `sim speed ~1.0x real time`.

**c. The full demo (webcam + simulator):**
```bash
ls /dev/video*                          # the webcam must show up here
# terminal 1
python3 sim_standing.py --rl-lab
# terminal 2, once the band is released
python3 leader_pose.py                  # click yourself in the camera window
# or record the proof video (rendered after you quit with 'q'):
python3 leader_pose.py --record-demo demo.mp4
```
With one person in view you can skip the click: `--num-people 1`.

## 11. If something goes wrong

| Symptom | Fix |
| --- | --- |
| `pip install -e unitree_sdk2_python` fails about CycloneDDS | `echo $CYCLONEDDS_HOME` must print the install folder from step 3; open a new terminal or `source ~/.bashrc`. |
| `Could not open input '0'` | No webcam: `ls /dev/video*`. In VirtualBox: Devices -> Webcams. |
| `No rt/lowstate received` | Start `sim_standing.py --rl-lab` first; `DOMAIN_ID = 1` and `INTERFACE = "lo"` in config.py. |
| `sim speed 0.3x real time` or lower, robot moves in slow motion | Not enough CPU: see the VirtualBox settings in section 0, close other programs, try `--viewer-fps 5`. |
| Robot falls | Keep `--arm-speed 1.0` (default); fast two-arm moves can topple the balance policy. Press `9` in the simulator window to re-attach the band. |
| `Model not found` / pose model error | Step 7; or pass `--model /path/to/pose_landmarker.task`. |
| Black / empty simulator window, GL errors | `sudo apt install libgl1 libegl1`; in a VM enable 3D acceleration. Headless machine: `python3 sim_standing.py --rl-lab --headless`. |

More in [RUNNING.md](RUNNING.md) (all options) and
[INTEGRATION.md](INTEGRATION.md) (how this fits with the gesture/walking
code and the real robot).
