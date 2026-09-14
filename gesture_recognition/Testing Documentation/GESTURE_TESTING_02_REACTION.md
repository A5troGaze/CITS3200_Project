# Gesture Reaction Formal Testing

## Purpose

The purpose of this test was to verify whether `react_to_gestures.py` can take gesture commands from recorded MP4 videos and the live camera feed and trigger simulated reactions corresponding to the gestures in MuJoCo.

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


### Test Procedure
---
Each of the eight gesture videos was tested individually using:

`python react_to_gestures.py --video <video_path>`

For each test, the recognised gesture and distance were observed in the terminal, while the corresponding G1 reaction was observed in the MuJoCo viewer.


### Test Results
---
| Gesture | Recognition | Actual Reaction | Result | Runtime Issue |
|---|---|---|---|---|
| `move_left` | Correct (2.68) | Left leg moved inward and the body moved slightly to the left. Clear visible response. | SUCCESS | Segmentation fault after test |
| `move_right` | Initially detected as `turn_right`, then correctly detected as `move_right` (2.35) | Clear response opposite to `move_left`. | SUCCESS | None observed |
| `turn_left` | Correct (2.49) | Clear and noticeable simulated model response. | SUCCESS | Segmentation fault after test |
| `turn_right` | Correct (1.90) | Body moved slightly to the right and the right leg moved outward. | SUCCESS | None observed |
| `move_forward_left_hand` | Correct (2.08) | Left arm raised clearly. | SUCCESS | None observed |
| `move_backward_left_hand` | Not recognised | No reaction triggered because the gesture was not recognised. | FAILURE | Segmentation fault after test |
| `move_forward_right_hand` | Correct (2.50) | Right arm raised clearly. | SUCCESS | None observed |
| `move_backward_right_hand` | Recognised, but unstable (approximately 2.95–2.98) | Right arm moved backward clearly. | SUCCESS | Segmentation fault after test |


### Observations
---
The updated gesture reactions were generally clearer and easier to observe in the MuJoCo simulation. Seven of the eight recorded gesture videos were recognised and produced visible simulated responses.

`move_backward_left_hand` was not successfully recognised during this round of testing, so its corresponding simulation reaction could not be verified.

`move_backward_right_hand` was recognised, but the recognition distance remained close to the classification threshold (approximately 2.95–2.98), indicating that recognition of the backward gesture was less stable.

Several test runs still ended with `Segmentation fault (core dumped)`. This occurred after testing `move_left`, `turn_left`, `move_backward_left_hand`, and `move_backward_right_hand`. The simulation stability issue therefore remains present.


### Conclusion
---
The updated testing confirmed that the recorded-video gesture recognition pipeline can successfully trigger visible reactions in the MuJoCo G1 simulation. Seven of the eight tested gestures were recognised and produced clear simulated responses.

The updated gesture reaction values made the simulated movements easier to observe compared with the previous testing. However, recognition of the backward gestures remains less reliable, particularly `move_backward_left_hand`, which was not recognised during this test.

Intermittent segmentation faults were also still observed during several test runs. Further investigation is required to determine the cause of these crashes.



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