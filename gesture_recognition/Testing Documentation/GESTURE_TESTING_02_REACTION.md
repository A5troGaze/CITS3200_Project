# Gesture Reaction Formal Testing

## Purpose

The purpose of this test was to verify whether `react_to_gestures.py` can will take gesture commands from recorded MP4 videos and the live camera feed and trigger simulated reactions corresponding to the gestures in MuJoCo.

The test focuses on the complete pipeline:

MP4 video → Gesture Recognition → Gesture Action Mapping → MuJoCo G1 Reaction

## Video File Input

### Test Environment
---
- Ubuntu 22 VM
- Python virtual environment: `g1-env`
- MediaPipe Hand Landmarker
- MuJoCo G1 simulation (`scene_with_hands.xml`)
- Input: Pre-recorded MP4 gesture videos
- Gesture classification threshold: 3.0
- Branch: `Simulation`

For turn gestures, `waist_yaw_joint` was used during testing to provide a visible turning reaction in the simulation.

### Test Procedure
---
Each of the eight gesture videos was tested individually using:

`python react_to_gestures.py --video <video_path>`

For each test, the recognised gesture and distance were observed in the terminal, while the corresponding G1 reaction was observed in the MuJoCo viewer.


### Test Results
---
| Gesture | Recognition | Expected Reaction | Actual Reaction | Result |
|---|---|---|---|---|
| `move_left` | Correct (2.68) | Robot reacts/moves left | No obvious robot movement | FAIL |
| `move_right` | Initially detected as `turn_right`, then correctly detected as `move_right` (2.35) | Robot reacts/moves right | Small rightward movement of the upper body and arms | NEEDS IMPROVEMENT |
| `turn_left` | Correct (2.49) | Robot turns/reacts left | Small leftward body movement | NEEDS IMPROVEMENT |
| `turn_right` | Correct (1.90) | Robot turns/reacts right | Small rightward body movement | NEEDS IMPROVEMENT |
| `move_forward_left_hand` | Correct (2.08) | Left arm moves forward | Left arm moved backward | FAIL |
| `move_backward_left_hand` | Test interrupted | Left arm moves backward | No reaction observed because the program aborted | TEST INTERRUPTED |
| `move_forward_right_hand` | Correct (2.50) | Right arm moves forward | Right arm moved backward | FAIL |
| `move_backward_right_hand` | Correct (approximately 2.95–2.98) | Right arm moves backward | Right arm moved forward/upward | FAIL |
---
### Observations
---
The gesture recognition component successfully recognised most of the test gestures. However, some recognition results were close to the classification threshold, particularly the backward hand gestures.

The MuJoCo simulation successfully demonstrated that recognised gestures can trigger robot reactions. However, several issues were identified:

- `move_left` did not produce an obvious robot reaction.
- `move_right`, `turn_left`, and `turn_right` produced visible but relatively small body movements.
- The forward and backward arm actions appear to use reversed actuator directions. Both forward hand gestures caused the corresponding arm to move backward, while `move_backward_right_hand` caused the right arm to move forward/upward.
- `move_backward_left_hand` could not be fully tested because the program aborted before a valid reaction was observed.
- Several test runs ended with errors such as `Segmentation fault (core dumped)` or `Aborted (core dumped)`, indicating a stability issue in the current simulation/testing environment.

### Conclusion
---
The testing confirmed that the complete pipeline from recorded video input to MuJoCo robot reaction is functioning. Most gestures can be recognised and can trigger a response from the simulated G1 robot.

However, the current gesture-to-action mappings still require improvement. In particular, the movement directions for the arm gestures should be corrected, and the reactions for left/right movement and turning should be made clearer. The program stability issue observed during several test runs should also be investigated.

Further changes and retesting are required before the gesture reactions can be considered reliable.



## Live Camera Input

### Testing Environment
---
- Operating System: `Ubuntu 22 (Physical device)`
- Input method: `Live camera feed`
- Gesture Recognition: `MediaPipe Hand Landmarker`
- Classification Threshold: `1.5`
- Simulated actuator velocity for testing: `1.5`
- Gesture Reference Data: `data/gestures.json`
- Simulation environment: `Mujoco`

### Test Procedure
---
1. Launch program
2. Hold the target gesture steadily for approximately 7 - 10 seconds.
3. Watch specified reaction take place in Mujoco simulation and take note whether the expected actuator defined in `GESTURE_ACTIONS` is moving.

### Test Results
---
| Gesture | Recognition | Expected Reaction | Actual Reaction | Result |
|---|---|---|---|---|
| `move_left` | Correct (1.27) | **Left hip roll actuator** turns _inward_ | Simulated G1 **left hip roll actuator** turned inward | SUCCESS |
| `move_right` | Correct (1.13) | **Right hip roll actuator** turns _inward_  | Simulated G1 **right hip roll actuator** turned inward | SUCCESS |
| `turn_left` | Correct (1.45) | **Left hip yaw actuator** turns _inward_  | Simulated G1 **left hip yaw actuator** turned inward | SUCCESS |
| `turn_right` | Correct (1.09) | **Right hip yaw actuator** moves _inward_ | Simulated G1 **right hip yaw actuator** turned inward | SUCCESS |
| `move_forward_left_hand` | Correct (0.69) | **Left shoulder pitch actuator** raises left hand upward | Simulated G1 **left shoulder pitch actuator** raised left arm upward | SUCCESS |
| `move_backward_left_hand` | Registered as `move_forward_right_hand` for a split second then fixed on `move_backward_left_hand` | **Left shoulder pitch actuator** sends arm backward | Simulated G1 **left shoulder pitch actuator** sent arm backward **AND** **right shoulder pitch actuator** raised right arm upward | **FAILURE** |
| `move_forward_right_hand` | Correct (1.01) | **Right shoulder pitch actuator** raises right hand upward | Simulated G1 **right shoulder pitch actuator** raised right arm upward | SUCCESS |
| `move_backward_right_hand` | Registered as `move_forward_left_hand` for a split second then fixed on `move_backward_right_hand` | **Right shoulder pitch actuator** sends arm backward | Simulated G1 **right shoulder pitch actuator** sent arm backward **AND** **left shoulder pitch actuator** raised right arm upward | **FAILURE** |
---
### Observations
---
One gesture being mistaken for another causes another action to start up and run. Causes two commands to be run. Should not work that way. Same problem as previous round of testing. Suggested fix has not been implemented yet.

### Conclusion
---
Testing proved that the correct command was being registered via hand signals to the simulation. Same problem as last time persisted as the suggested fix had not been implemented yet. Suggested fix should be implemented before next phase. 