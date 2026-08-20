# Gesture Recognition Formal Testing

## Testing Objective

The purpose of this testing is to evaluate whether the current gesture recognition system can correctly identify the predefined gestures using recorded video input.

A temporary classification threshold of 5.0 was used for this stage of functional testing. The threshold was increased from 0.9 to verify that the classification pipeline could recognise and return gesture classes. Practical threshold tuning will be performed at a later stage.

## Testing Environment

- Operating System: Ubuntu 22 VM (VMware)
- Python Environment: g1-env
- Input Method: Pre-recorded MP4 videos
- Gesture Recognition: MediaPipe Hand Landmarker
- Classification Threshold: 5.0
- Gesture Reference Data: data/gestures.json

Live webcam testing was not performed because the Ubuntu VM did not expose a video device (/dev/video*). Pre-recorded videos were used as the input instead.

## Test Procedure

1. Refer to the corresponding gesture image in the gesture library.
2. Record a separate video for each gesture.
3. Hold the target gesture steadily for approximately 7 - 10 seconds.
4. Transfer the recorded video to the Ubuntu VM.
5. Run the video through the gesture recognition program using the --video option.
6. Record the expected gesture, actual recognition result, closest distance, and any observed issues.

## Test Results

| Test ID | Gesture | Expected | Actual | Closest Distance | Threshold | Result |
|---|---|---|---|---:|---:|---|
| T01 | Move Left | move_left | move_left | 2.672 | 5.0 | Pass |
| T02 | Move Right | move_right | move_right | 2.307 | 5.0 | Pass |
| T03 | Turn Left | turn_left | turn_left | 1.946 | 5.0 | Pass |
| T04 | Turn Right | turn_right | turn_right | 1.766 | 5.0 | Pass |
| T05 | Move Forward - Left Hand | move_forward_left_hand | move_forward_left_hand | 2.075 | 5.0 | Pass |
| T06 | Move Forward - Right Hand | move_forward_right_hand | move_forward_right_hand / move_backward_left_hand | 2.199 | 5.0 | Fail / Unstable |
| T07 | Move Backward - Left Hand | move_backward_left_hand | move_backward_left_hand | 3.077 | 5.0 | Pass |
| T08 | Move Backward - Right Hand | move_backward_right_hand | move_backward_right_hand | 3.291 | 5.0 | Pass |

## Observations

- T01: move_left was consistently recognised correctly.
- T02: move_right was consistently recognised after the hand position stabilised. A brief turn_right classification occurred at the beginning of the video.
- T03: turn_left was consistently recognised correctly.
- T04: turn_right was consistently recognised correctly.
- T05: move_forward_left_hand was consistently recognised correctly, although the distance values varied more during the video.
- T06: move_forward_right_hand showed unstable classification and frequently switched between move_forward_right_hand and move_backward_left_hand.
- T07: move_backward_left_hand was consistently recognised correctly.
- T08: move_backward_right_hand was consistently recognised correctly.

## Issues Identified

- move_forward_right_hand was frequently confused with move_backward_left_hand during T06.
- A brief incorrect classification occurred at the beginning of the move_right test while the hand position was stabilising.
- Live webcam testing could not be performed in the current Ubuntu VM environment.
- The threshold of 5.0 is temporary and is used for functional testing only.

## First-Round Testing Summary

Eight predefined gesture classes were tested using separate pre-recorded videos with a temporary classification threshold of 5.0.

Seven of the eight first-round functional test cases passed. The move_forward_right_hand test showed unstable classification and was frequently confused with move_backward_left_hand.

Further repeat testing is required to determine whether this issue is consistently reproducible. Practical threshold tuning and testing under different camera conditions will be performed at a later stage.
