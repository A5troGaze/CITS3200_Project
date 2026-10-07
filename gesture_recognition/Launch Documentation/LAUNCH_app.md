# Running `gesture_control.py` — Quick Start

This is the quickstart guide to launch the program at [07/09/2026]. Steps outline three different Terminals running three different items. **Terminal 1** runs the program. **Terminal 2** runs the mujoco simulation and receives CycloneDDS commands. **Terminal 3** runs a walking policy for controlling the robot.

In **Gesture** mode **Terminal 1** sends gamepad commands to the walking policy based on recognised hand gestures. The walking policy in **Terminal 2** in turn maintains balance in the simulation and translates the gamepad instructions into low level commands to send to the Mujoco simulation running in **Terminal 3**

In **Mimic** mode

# Startup Sequence

## Terminal 1:
Activate the venv:
```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```
Start up gesture controller:
```bash
cd ~/CITS3200/Project/gesture_recognition
```
```bash
python gesture_control.py [sim|real]
```

## Terminal 2:
Activate the venv:
```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```
Start up C++ built Mujoco:
```bash
cd ~/CITS3200/Dependencies/unitree_mujoco/simulate/build
```
```bash
./unitree_mujoco
```

## Terminal 3:
Activate the venv:
```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```
Start up rl_lab's C++ built walking-policy controller:
```bash
cd ~/CITS3200/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof/build
```
```bash
./g1_ctrl --network lo
```

# Usage Instructions:
