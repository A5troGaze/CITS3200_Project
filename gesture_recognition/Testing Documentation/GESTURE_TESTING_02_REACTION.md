# Gesture Reaction Formal Testing

## Purpose

The purpose of this test was to verify whether `react_to_gestures.py` can recognise gesture commands from recorded MP4 videos and trigger the corresponding reactions in the MuJoCo G1 simulation.

The test focuses on the complete pipeline:

MP4 video → Gesture Recognition → Gesture Action Mapping → MuJoCo G1 Reaction

## Test Environment

- Ubuntu 22 VM
- Python virtual environment: `g1-env`
- MediaPipe Hand Landmarker
- MuJoCo G1 simulation (`scene_with_hands.xml`)
- Input: Pre-recorded MP4 gesture videos
- Gesture classification threshold: 3.0
- Branch: `Simulation`

For turn gestures, `waist_yaw_joint` was used during testing to provide a visible turning reaction in the simulation.

## Test Method

Each of the eight gesture videos was tested individually using:

`python react_to_gestures.py --video <video_path>`

For each test, the recognised gesture and distance were observed in the terminal, while the corresponding G1 reaction was observed in the MuJoCo viewer.

## Test Results

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

## Observations

The gesture recognition component successfully recognised most of the test gestures. However, some recognition results were close to the classification threshold, particularly the backward hand gestures.

The MuJoCo simulation successfully demonstrated that recognised gestures can trigger robot reactions. However, several issues were identified:

- `move_left` did not produce an obvious robot reaction.
- `move_right`, `turn_left`, and `turn_right` produced visible but relatively small body movements.
- The forward and backward arm actions appear to use reversed actuator directions. Both forward hand gestures caused the corresponding arm to move backward, while `move_backward_right_hand` caused the right arm to move forward/upward.
- `move_backward_left_hand` could not be fully tested because the program aborted before a valid reaction was observed.
- Several test runs ended with errors such as `Segmentation fault (core dumped)` or `Aborted (core dumped)`, indicating a stability issue in the current simulation/testing environment.

## Conclusion

The testing confirmed that the complete pipeline from recorded video input to MuJoCo robot reaction is functioning. Most gestures can be recognised and can trigger a response from the simulated G1 robot.

However, the current gesture-to-action mappings still require improvement. In particular, the movement directions for the arm gestures should be corrected, and the reactions for left/right movement and turning should be made clearer. The program stability issue observed during several test runs should also be investigated.

Further changes and retesting are required before the gesture reactions can be considered reliable.
