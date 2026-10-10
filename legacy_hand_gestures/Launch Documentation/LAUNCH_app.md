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
---


# Usage Instructions:
## Step 1: Follow instructions in Terminal 1
Terminal 1 will prompt you through a series of actions in order to prepare the simulation and start the policy:
- `Enter` to begin and put the robot into standing position
- `Enter` to ground its feet on the floor of the simulation
- `d` to confirm the feet are grounded
- `Enter` to start running the policy
- `Enter` to release the elastic band and launch the Gesture mode

## Step 2: Gesture Mode
You will see two windows, **Mujoco** and the **Camera Window**. When a hand enters the camer window's frame a skeleton will be drawn ontop. In order to use:
- Make a recognised gesture
- The command associated to that gesture will be sent to Mujoco and the simulated robot will respond.
- Stop holding a recognised gesture and the robot with stop moving
- Press the `m` key and Mimic mode will be launched
- Press the `q` key and the program will stop running

## Step 3: Mimic Mode
