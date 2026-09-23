# Integrating person_id with the gesture / walking code

Status: proposal. Nothing in the gesture or walking code has been changed.
It was read on the branches `Walking-Policy-Test`, `New-Gestures` and `Simulation`
(Chris), and `CycloneDDS-Simulation` / `Camera-Source` (Mengfei), as of
2026-09-23.

## Who publishes what today

| Component | Sim (unitree_mujoco, domain 1, `lo`) | Real robot (domain 0) |
| --- | --- | --- |
| person_id (`mujoco_pose_controller.py`) | `rt/lowcmd`, all 29 motors at 200 Hz. Arms + waist tracked; legs held (default) or limp (`--commanded-only`) | `rt/arm_sdk`, waist + arms only, weight in `motor_cmd[29].q` (`arm_sdk_publisher.py`, `--real`) |
| Gesture reaction (`simulation_controller.SimController`) | `rt/lowcmd`, all 29 motors at 500 Hz, holding a home pose plus the gesture's joint offsets | `real_controller.RealController`: `LocoClient.Move(vx, vy, vyaw)`, Unitree's own locomotion |
| Walking policy (`rl_lab_walking_test.py`: unitree_rl_lab's whole-body ONNX velocity policy; `gesture_to_vgamepad.py` drives the C++ deploy through a virtual pad) | `rt/lowcmd`, all 29 motors **including the arms**, which are part of the policy's output | n/a (LocoClient above) |

## Why two publishers fight in the sim

unitree_mujoco's `LowCmdHandler` runs once per received `rt/lowcmd` message
and rewrites `ctrl` for **all 29 actuators** from that message alone. It
keeps that torque until the next message arrives, from any publisher. With
two publishers running (person_id + SimController, or person_id + the
walking policy), the robot receives alternating complete commands 200-500
times a second: each one zeros or overrides the other's joints. The result
is shaking, and the walking policy loses its leg torques half of the time.
person_id's `--commanded-only` does not fix this: kp = kd = 0 on the legs
still sets their torque to zero in every person_id message.

**Rule for the sim: exactly one process publishes `rt/lowcmd`.**

A second trap: when a publisher stops, the sim keeps applying its last
torque indefinitely. person_id therefore sends a zero-torque release when it
stops. A mode switch must hand over without a gap, or with a release.

## On the real robot they can coexist

On the G1, locomotion runs in Unitree's firmware and is driven by
`LocoClient`. `rt/arm_sdk` blends a separate command into the arms and
waist only, with the blend weight in `motor_cmd[29].q`. The legs stay under
the balance controller. So on hardware:

* gesture mode = `LocoClient.Move` (walk/turn), arm_sdk weight 0;
* person_id mode = `LocoClient` standing still, arm_sdk weight ramped to 1;
* both at once (walk by gesture while the arms mimic) is possible later.
  It needs a way to give walking commands without the hand gestures that
  are being mimicked.

`rt/lowcmd` must never be published on the real robot by either component
in normal operation: it overrides everything, including balance.

## Proposed switch: one owner process with a mode flag

The client asked for a switch between person-ID mode and gesture mode.
Proposal: a small **mode manager** that owns the camera and the one DDS
publisher, and runs one "brain" at a time.

```
camera ──► mode_manager ──► person_id brain (MimicPipeline)   ─┐
                 │          gesture brain (gesture_core + map) ─┤► one output
                 │                                              │   sim:  single rt/lowcmd publisher
          mode = PERSON_ID | GESTURE | IDLE                     │   real: arm_sdk + LocoClient
          switch: key 'm', CLI, or a held "switch" gesture  ────┘
```

Interface each brain implements (a plain Python protocol, no DDS inside):

```python
class Brain:
    def step(self, frame_bgr, t) -> "BrainOutput": ...
    def reset(self) -> None: ...

@dataclass
class BrainOutput:
    joint_targets: dict[int, float]            # motor index -> rad (arms/waist); may be empty
    base_velocity: tuple[float, float, float] | None = None   # (vx, vy, vyaw) for locomotion
```

* person_id brain: `MimicPipeline.step(landmarks, t).targets` -> `joint_targets`;
  `base_velocity = (0, 0, 0)`.
* gesture brain: recognised gesture -> `base_velocity` (real robot and
  walking policy), or -> `joint_targets` (the current sim demo's
  `GESTURE_TARGETS`).
* The owner routes the output:
  * sim: one publisher builds each LowCmd from the active brain's
    `joint_targets` (legs from the walking policy, or held). The walking
    policy's actions would be read in-process rather than published
    separately, so there is still only one `rt/lowcmd` source;
  * real: `joint_targets` -> `ArmSdkPublisher.set_targets`;
    `base_velocity` -> `LocoClient.Move`.
* Switching: the arms ease to neutral (arm_sdk weight ramps to 0 when
  leaving person_id mode), the new brain is `reset()`, then it starts.
  Leaving person_id mode is `ArmSdkPublisher.disengage()` / `clear_targets()`.
* Both brains can share a single MediaPipe pass: the gesture code uses the
  HandLandmarker and person_id the PoseLandmarker, on the same frame.
* The switch gesture should be one the person_id leader won't do by
  accident while mimicking (e.g. both hands on head for 2 s), or just a key.

What each side would need to change (for the team to decide, not done here):
* gesture code: expose a `Brain`-style `step()` that returns targets/velocity
  instead of owning a publisher thread (its `set_gesture()` already
  separates recognition from publishing);
* person_id: `leader_pose.py`'s per-frame body becomes `PersonIdBrain.step()`.
  `MimicPipeline` and the publishers are already DDS-free or lazily
  imported, so this is a refactor, not a rewrite.

## Open questions for the team

1. Does the gesture sim demo keep publishing `rt/lowcmd` itself, or move to
   the walking policy + virtual gamepad (Walking-Policy-Test)? The owner
   process looks different for each. The walking policy was trained with its
   own arm motion, so overriding its arms with person_id's while it walks
   takes it outside its training (Chris's SETUP_rl_lab_walking_test.md
   notes the same). In the sim, mimic while standing and walk while not
   mimicking.
2. Is the real G1 a waist-locked unit? If so, run person_id with `--waist yaw`.
3. Walk and mimic at the same time on hardware, or strictly one mode at a time?
