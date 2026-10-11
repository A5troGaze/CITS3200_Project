# Gesture Recognition Formal Testing

## Testing Objective

The purpose of this testing is to evaluate whether the current gesture recognition system can correctly identify the predefined gestures using recorded video input.

A temporary classification threshold of 5.0 was used for Video File Testing. The threshold was increased from 1.5 to verify that the classification pipeline could recognise and return gesture classes. For live testing the threshold stayed at 1.5.Practical threshold tuning will be performed at a later stage.

---
## Video File Input

### Testing Environment
---
- Operating System: `Ubuntu 22 VM (VMware)`
- Input Method: `Pre-recorded MP4 videos`
- Gesture Recognition: `MediaPipe Hand Landmarker`
- Classification Threshold: `5.0`
- Gesture Reference Data: `data/gestures.json`

### Test Procedure
---
1. Refer to the corresponding gesture image in the gesture library.
2. Record a separate video for each gesture.
3. Hold the target gesture steadily for approximately 7 - 10 seconds.
4. Transfer the recorded video to the Ubuntu VM.
5. Run the video through the gesture recognition program using the --video option.
6. Record the expected gesture, actual recognition result, closest distance, and any observed issues.

### Test Results
---
| Test ID | Gesture | Expected | Actual | Closest Distance | Threshold | Result |
|---|---|---|---|---:|---:|---|
| T01 | Move Left | `move_left` | `move_left` | 2.672 | 5.0 | Pass |
| T02 | Move Right | `move_right` | `move_right` | 2.307 | 5.0 | Pass |
| T03 | Turn Left | `turn_left` | `turn_left` | 1.946 | 5.0 | Pass |
| T04 | Turn Right | `turn_right` | `turn_right` | 1.766 | 5.0 | Pass |
| T05 | Move Forward - Left Hand | `move_forward_left_hand` | `move_forward_left_hand` | 2.075 | 5.0 | Pass |
| T06 | Move Forward - Right Hand | `move_forward_right_hand` | `move_forward_right_hand` / `move_backward_left_hand` | 2.199 | 5.0 | Fail / Unstable |
| T07 | Move Backward - Left Hand | `move_backward_left_hand` | `move_backward_left_hand` | 3.077 | 5.0 | Pass |
| T08 | Move Backward - Right Hand | `move_backward_right_hand` | `move_backward_right_hand` | 3.291 | 5.0 | Pass |

### Observations
---
- T01: `move_left` was consistently recognised correctly.
- T02: `move_right` was consistently recognised after the hand position stabilised. A brief `turn_right` classification occurred at the beginning of the video.
- T03: `turn_left` was consistently recognised correctly.
- T04: `turn_right` was consistently recognised correctly.
- T05: `move_forward_left_hand` was consistently recognised correctly, although the distance values varied more during the video.
- T06: `move_forward_right_hand` showed unstable classification and frequently switched between `move_forward_right_hand` and `move_backward_left_hand`.
- T07: `move_backward_left_hand` was consistently recognised correctly.
- T08: `move_backward_right_hand` was consistently recognised correctly.

### Issues Identified
---
- `move_forward_right_hand` was frequently confused with `move_backward_left_hand` during T06.
- A brief incorrect classification occurred at the beginning of the `move_right` test while the hand position was stabilising.
- The threshold of 5.0 is temporary and is used for functional testing only.

### Video Input Testing Summary
---
Eight predefined gesture classes were tested using separate pre-recorded videos with a temporary classification threshold of 5.0.

Seven of the eight first-round functional test cases passed. The `move_forward_right_hand` test showed unstable classification and was frequently confused with `move_backward_left_hand`.

Further repeat testing is required to determine whether this issue is consistently reproducible. Practical threshold tuning and testing will be performed at a later stage.

---


## Live Camera Input

### Testing Environment
---
- Operating System: `Ubuntu 22 (Physical device)`
- Input method: `Live camera feed`
- Gesture Recognition: `MediaPipe Hand Landmarker`
- Classification Threshold: `1.5`
- Gesture Reference Data: `data/gestures.json`

### Test Procedure
---
1. Launch program
2. Hold the target gesture steadily for approximately 7 - 10 seconds.
3. Record the expected gesture, actual recognition result, closest distance, and any observed issues.

### Test Results
---
| Test ID | Gesture | Expected | Actual | Closest Distance | Threshold | Result |
|---|---|---|---|---:|---:|---|
| T09 | Move Left | `move_left` | `move_left` | 0.883 | 1.5 | Pass |
| T10 | Move Right | `move_right` | `move_right` | 0.533 | 1.5 | Pass |
| T11 | Turn Left | `turn_left` | `turn_left` | 0.390 | 1.5 | Pass |
| T12 | Turn Right | `turn_right` | `turn_right` | 0.476 | 1.5 | Pass |
| T13 | Move Forward - Left Hand | `move_forward_left_hand` | `move_forward_left_hand` | 0.358 | 1.5 | Pass |
| T14 | Move Forward - Right Hand | `move_forward_right_hand` | `move_forward_right_hand` | 0.859 | 1.5 | Pass |
| T15 | Move Backward - Left Hand | `move_backward_left_hand` | `move_backward_left_hand` / `move_forward_right_hand` | 0.298 | 1.5 | Fail |
| T16 | Move Backward - Right Hand | `move_backward_right_hand` | `move_backward_right_hand` | 0.363 | 1.5 | Pass |
---
### Observations
---
- T09: `move_left` was consistently recognised correctly.
- T10: `move_right` was consistently recognised correctly.
- T11: `turn_left` was consistently recognised correctly.
- T12: `turn_right` was consistently recognised correctly.
- T13: `move_forward_left_hand` was consistently recognised correctly.
- T14: `move_forward_right_hand` was consistently recognised correctly
- T15: `move_backward_left_hand` was recognised correctly after stabilising in frame. `move_backward_left_hand` would be briefly classified as `move_forward_right_hand`.
- T16: `move_backward_right_hand` was consistently recognised correctly.

### Issues Identified
---
- `move_backward_left_hand` was frequently confused with `move_forward_right_hand` during T15.

### Video Input Testing Summary
---
Testing proved that the majority of the hand signals remained stable. However there is an issue with `move_backward_left_hand` and `move_forward_right_hand`. I believe this is due to the flattened vectors being rather similar. A solution would be to differ the signals for forward and backward to ensure that the vectors are dissimilar enough.