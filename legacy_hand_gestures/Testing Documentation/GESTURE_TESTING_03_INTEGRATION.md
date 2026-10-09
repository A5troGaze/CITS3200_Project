# Integration Testing: Gesture + Person ID Mode Switch

## Purpose

`gesture_control.py` now runs two modes on one camera. Pressing `m` switches between them:

- **gesture mode**: hand gestures walk the robot (the existing pipeline).
- **mimic mode**: person_id (`PersonIdBrain`) tracks the person in front of the camera and produces arm joint targets.

This document is for whoever does the final test of that switch. It says how to run it, what should happen, and what is known not to work yet, so you don't log known gaps as new bugs.

Branch: `Integration` (until it is merged into `main`).

## What is and isn't connected

Connected:
- The `m` key, the mode switch, and the shared camera. `gesture_control.py` owns the camera and window. person_id does not open its own.
- In the sim, switching to mimic clears the current gesture and pauses `g1_ctrl` (`SIGSTOP`). Switching back resumes it (`SIGCONT`).
- `PersonIdBrain` runs on every frame in mimic mode and returns joint targets.

**Not connected: the joint targets are not sent to the robot.** In mimic mode the robot's arms will not move. The stock `unitree_mujoco` + `g1_ctrl` setup doesn't read `rt/arm_sdk`, and pausing `g1_ctrl` also pauses the balance controller (see `person_id/INTEGRATION.md`). How to send them is still undecided. So for now, "mimic mode works" means the brain runs and produces targets, not that the robot copies you.

This document covers the simulation only. The real robot is not part of this test.

## Test Environment

Fill this in when you test.

- OS / machine:
- Python environment (`g1-env`?):
- Camera:
- Branch and commit (`git log -1 --oneline`):

## Before you start

1. Follow `SETUP_GUIDE_V1.md` to get `unitree_mujoco` and `g1_ctrl` working. Hand gestures should already walk the robot in the sim before you test the switch.
2. You need the pose model in addition to the hand model. person_id looks for `~/CITS3200/Dependencies/Models/pose_landmarker.task` (or the folder in the `CITS3200_MODELS_DIR` environment variable). The download command is in `person_id/INSTALL.md`, step 7.
3. The model paths in the code still say `~/CITS3200/...`, while `SETUP_GUIDE_V1.md` uses `~/HumanoidControl/...`. If you followed the guide, put the models in the `~/CITS3200/Dependencies/Models/` folder the code expects, or the program won't start.
4. Optional check that person_id works on its own, with no sim and no robot:
   `python3 person_id/leader_pose.py --dry-run`
   It prints joint targets for the person in front of the camera.
5. Optional unit tests: `python3 -m pytest person_id/tests/test_person_id_brain.py` (4 tests, no camera or model needed).

## Test Procedure

Run from the `gesture_recognition` folder:

`python gesture_control.py sim`

Start `unitree_mujoco` and `g1_ctrl` in their own terminals first, then follow the prompts in the terminal (stand up, ground the feet with key `8`, run the policy, release the band with key `9`). Keys in the camera window: `m` toggles mode, `q` quits.

### Test 1: gesture mode still works
Hold a walking gesture (for example `move_forward_right_hand`). The robot should walk. Expected: same as before the integration changes.

### Test 2: switch to mimic mode
1. Press `m` while a person is in view.
2. Terminal prints `switched to mimic mode`.
3. First time only: there is a short pause while the pose model loads.
4. The window text shows `Mimic mode (N joint targets)`.

Expected: N is 0 until the person is detected, then goes above 0. The robot should stop walking (the gesture is cleared). Note what the robot does after `g1_ctrl` is paused: does it stay standing, sag, or fall? We have not seen this in the sim.

### Test 3: switch back to gesture mode
Press `m` again. Terminal prints `switched to gesture mode`. Hold a walking gesture. Expected: `g1_ctrl` resumes and the robot walks again.

### Test 4: switch repeatedly
Press `m` at least 10 times, back and forth, including a few quick presses. Expected: no crash, no frozen window, gestures still work after the last switch to gesture mode.

### Test 5: leader tracking in mimic mode
In mimic mode:
- Stand in front of the camera and check N goes above 0.
- Walk out of frame. Expected: the targets hold for a moment, then ease back to neutral.
- Come back. Expected: it picks you up again.
- With two people in view, note who it locks onto. There is no click-to-select in this mode; it takes the first detected person.

### Test 6: left and right
The frame the brain gets is mirror-flipped by `gesture_control.py` and flipped back inside `PersonIdBrain`. Left and right should match the standalone `leader_pose.py`. This is the most likely place for a silent bug.

Right now `gesture_control.py` only shows the number of targets, not their values, so you can't compare it to `leader_pose.py --dry-run` (which prints the joint numbers) without a code change. If you want to check it, temporarily add `print(output.joint_targets)` after the `mimic_brain.step(...)` line in `gesture_control.py`, raise your left arm in both programs, and compare which joints move. Remove the print afterwards. Mark this test as skipped if you don't do it.

### Test 7: pose model missing
Temporarily rename `pose_landmarker.task`, start `gesture_control.py`, press `m`. Expected: the terminal prints `Couldn't start PersonIdBrain: ...` and `Mimic mode will do nothing.`, the window shows `Mimic mode (0 joint targets)`, and gesture mode still works after pressing `m` again. Rename the file back afterwards.

### Test 8: `g1_ctrl` not running
Stop `g1_ctrl`, then press `m`. The program is expected to raise `g1_ctrl isn't running -- can't mute/unmute it.` Note whether this crashes the whole program. We know about it and haven't decided if it should be handled more gently.

## Test Results

| Test | Expected | Actual | Result |
|---|---|---|---|
| 1. Gesture mode walks | Robot walks | | |
| 2. Switch to mimic | N joint targets > 0 once detected, robot stops walking | | |
| 3. Switch back | `g1_ctrl` resumes, robot walks | | |
| 4. Switch 10+ times | No crash | | |
| 5. Leader tracking | Hold, ease to neutral, re-acquire | | |
| 6. Left and right | Matches `leader_pose.py` | | |
| 7. Pose model missing | Error printed, gesture mode still works | | |
| 8. `g1_ctrl` not running | RuntimeError message | | |

## Observations

(Write what you saw here, especially what the robot does while `g1_ctrl` is paused.)

## Known issues going in

- Joint targets are not sent to the robot, so mimic mode does not move the arms (see above).
- No click-to-select leader in mimic mode.
- Model paths are hardcoded to `~/CITS3200/...` while `SETUP_GUIDE_V1.md` uses `~/HumanoidControl/...`. The folder name is meant to be renamed later.
- The first press of `m` takes a moment because the pose model loads then.
