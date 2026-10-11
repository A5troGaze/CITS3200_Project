# G1 conventions used by person_id

Everything below was read from source or measured by loading the models in
MuJoCo, not assumed. Versions checked (2026-09-23):

| Dependency | Location | Commit / version |
| --- | --- | --- |
| GMR | `~/CITS3200/Dependencies/GMR` | `bb1bbe4` (general_motion_retargeting 0.2.0) |
| unitree_mujoco | `~/CITS3200/Dependencies/unitree_mujoco` | `1eb6642` |
| unitree_sdk2_python | `~/CITS3200/Dependencies/unitree_sdk2_python` | `65691c8` |
| Python (g1-env) | | 3.10.12 (GMR needs >= 3.10) |
| mujoco / mink / mediapipe | g1-env | 3.11.0 / 1.2.0 / 1.0.0 |

GMR's dependencies were already installed in g1-env with no conflicts; no new
environment was needed. The `smplx`/`torch` packages GMR pulls in are only used
by its SMPL-X file loaders, which person_id does not call.

## 1. Motor order (LowCmd `motor_cmd[i]` / LowState `motor_state[i]`)

Source: `unitree_mujoco/unitree_robots/g1/g1_joint_index_dds.md` (29-DoF table),
`unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py` (`G1JointIndex`),
and the sim itself: `UnitreeSdk2Bridge.LowCmdHandler` writes `motor_cmd[i]` to
`mj_data.ctrl[i]`, and actuator `i` of `g1_29dof.xml` drives the joint listed
below (checked by reading `actuator_trnid`).

| idx | joint (MJCF name) | idx | joint (MJCF name) |
| --- | --- | --- | --- |
| 0 | left_hip_pitch_joint | 15 | left_shoulder_pitch_joint |
| 1 | left_hip_roll_joint | 16 | left_shoulder_roll_joint |
| 2 | left_hip_yaw_joint | 17 | left_shoulder_yaw_joint |
| 3 | left_knee_joint | 18 | left_elbow_joint |
| 4 | left_ankle_pitch_joint | 19 | left_wrist_roll_joint |
| 5 | left_ankle_roll_joint | 20 | left_wrist_pitch_joint |
| 6 | right_hip_pitch_joint | 21 | left_wrist_yaw_joint |
| 7 | right_hip_roll_joint | 22 | right_shoulder_pitch_joint |
| 8 | right_hip_yaw_joint | 23 | right_shoulder_roll_joint |
| 9 | right_knee_joint | 24 | right_shoulder_yaw_joint |
| 10 | right_ankle_pitch_joint | 25 | right_elbow_joint |
| 11 | right_ankle_roll_joint | 26 | right_wrist_roll_joint |
| 12 | waist_yaw_joint | 27 | right_wrist_pitch_joint |
| 13 | waist_roll_joint | 28 | right_wrist_yaw_joint |
| 14 | waist_pitch_joint | | |

`motor_cmd[29]` is not a motor. On `rt/arm_sdk` its `.q` is the arm_sdk blend
weight (section 5).

The code never relies on this order being the same as some other list. The
mapping lives in `g1_joint_limits.py` (`G1_29DOF_JOINT_LIMITS[name].index`).
`gmr_retarget.py` reads each joint's qpos address from the GMR model by name.
`tests/test_g1_mapping.py` checks the table against both MJCF files.

## 2. GMR

### Per-frame API
`general_motion_retargeting/motion_retarget.py`, class `GeneralMotionRetargeting`:

* `GeneralMotionRetargeting(src_human, tgt_robot, actual_human_height=None,
  solver="daqp", damping=0.5, verbose=True, use_velocity_limit=False)`.
  Loads `ROBOT_XML_DICT[tgt_robot]` and `IK_CONFIG_DICT[src_human][tgt_robot]`
  (`params.py`). Both are plain module-level dicts. person_id adds its own
  entry (`IK_CONFIG_DICT["mediapipe"]["unitree_g1"]`) at runtime, pointing
  at `person_id/configs/mediapipe_to_g1.json`, so GMR's own files are
  not edited.
* `retarget(human_data) -> qpos` (copy of the internal mink configuration).
  The mink `Configuration` persists between calls, so every call warm-starts
  from the previous solution. Each call runs up to 1 + `max_iter` (=10)
  `mink.solve_ik` steps per match table and stops early when the task error
  improves by less than 0.001.

### Human-frame input format
`human_data` is a dict `{body_name: (pos, quat)}`:
* `pos`: 3-vector, metres, **world frame, z up** (SMPL-X/AMASS data is z-up).
* `quat`: global orientation as **scalar-first `(w, x, y, z)`**
  (`R.from_quat(..., scalar_first=True)` in `offset_human_data`).
* Body names are whatever the IK config's match tables and
  `human_scale_table` use (for SMPL-X: `pelvis, spine3, left_hip, left_knee,
  left_foot, left_shoulder, left_elbow, left_wrist`, plus the right-side
  equivalents).

Processing order in `update_targets`:
1. `scale_human_data`: each body's position relative to the root is scaled by
   `human_scale_table[body]`. Each table entry is first multiplied by
   `actual_human_height / human_height_assumption`. Bodies missing from the
   scale table are dropped.
2. `offset_human_data`: the target orientation is `q_human * rot_offset`, then
   `pos_offset` is applied in the rotated frame.
3. The ground offset is subtracted.

### SMPL-X -> unitree_g1 config (`ik_configs/smplx_to_g1.json`)
* Root: `pelvis` -> `pelvis`, `human_height_assumption` 1.8 m, ground 0.
* Scale: 0.9 for pelvis/spine/legs, 0.8 for arm bodies.
* Table 1 (position, rotation weights): pelvis (100, 10), toes (100, 10); every other
  body has position weight 0 and rotation weight 10. Table 2 then adds position weights
  (10 for knees, hips and arm joints) with rotation weights of 5.
* Robot frames used: `pelvis, left/right_hip_roll_link, left/right_knee_link,
  left/right_toe_link, torso_link, left/right_shoulder_yaw_link,
  left/right_elbow_link, left/right_wrist_yaw_link`.
* The rotation offsets convert SMPL-X joint frames to G1 link frames.

MediaPipe has no joint orientations, so person_id does not reuse this config.
It builds each target frame directly in the G1 link convention, so all
rotation offsets are identity (section 6).

### Model and output
* Model: `GMR/assets/unitree_g1/g1_mocap_29dof.xml`, nq = 36 (7 free-base +
  29 hinges), nv = 35. The output qpos is `[x y z qw qx qy qz, 29 joints]`.
  The 29 hinge joints have **the same names in the same order** as
  unitree_mujoco's `g1_29dof.xml` and the motor table above. Checked joint by joint.
* The zero-pose body positions of every arm and torso link are identical
  between the two models. The only difference is the torso/waist_roll body
  z (0.844 vs 0.835/0.854), which comes from a different split of the same chain.
* **Joint ranges differ.** The GMR model is tighter, presumably to avoid
  self-collision and extreme poses:

| joint | GMR g1_mocap | unitree_mujoco / real (URDF) |
| --- | --- | --- |
| hip pitch | [-1.57, 1.57] | [-2.5307, 2.8798] |
| L hip roll / R hip roll | [-0.5236, 1.57] / [-1.57, 0.5236] | [-0.5236, 2.9671] / [-2.9671, 0.5236] |
| hip yaw | [-1.57, 1.57] | [-2.7576, 2.7576] |
| waist yaw | [-1.57, 1.57] | [-2.618, 2.618] |
| shoulder pitch | [-3.0892, 1.149] | [-3.0892, 2.6704] |
| L / R shoulder roll | [-0.6, 2.2515] / [-2.2515, 0.6] | [-1.5882, 2.2515] / [-2.2515, 1.5882] |
| L / R shoulder yaw | [-1.4, 2.0] / [-2.0, 1.4] | [-2.618, 2.618] |
| elbow | [-1.0472, 1.7] | [-1.0472, 2.0944] |
| knee, ankles, waist roll/pitch, wrists | identical | identical |

  The IK runs inside GMR's tighter ranges. The output is then clamped again
  to the real limits in `g1_joint_limits.py` (from Unitree's
  `g1_29dof_rev_1_0.urdf`), which match `g1_29dof.xml`. So no command can
  leave the hardware range, and in practice nothing leaves GMR's safer range.

## 3. unitree_mujoco (`simulate_python/config.py`)

| setting | value |
| --- | --- |
| `ROBOT` / `ROBOT_SCENE` | `g1` / `../unitree_robots/g1/scene.xml` (includes `g1_29dof.xml`) |
| `DOMAIN_ID` / `INTERFACE` | `1` / `"lo"` |
| `ENABLE_ELASTIC_BAND` | `True`. The band pulls `torso_link` toward (0, 0, 3) with k = 200, d = 100. Keys: `9` toggles it, `7`/`8` shorten/lengthen it. |
| `SIMULATE_DT` / `VIEWER_DT` | 0.005 s (overrides the MJCF 0.002) / 0.02 s |

* Topics: the sim **subscribes only to `rt/lowcmd`** and publishes
  `rt/lowstate`, `rt/sportmodestate` and `rt/wirelesscontroller`. It does
  **not** subscribe to `rt/arm_sdk`, so the arm_sdk path can't be tested in this sim.
* PD law (`unitree_sdk2py_bridge.py`, `LowCmdHandler`), applied to all 29
  actuators **only when a LowCmd arrives**. `ctrl` holds its value between messages:
  `ctrl[i] = tau + kp*(q_des - q) + kd*(dq_des - dq)`. Actuators are torque
  motors clipped to their `ctrlrange`: ±25 N·m for shoulders/elbow/wrist roll,
  ±5 for wrist pitch/yaw, ±88 for waist yaw, ±50 for waist roll/pitch.
* `mode_pr`, `mode_machine`, `motor_cmd[i].mode` and the CRC are ignored by the sim.
  person_id still sets them the way the real robot needs.
* Arm joint damping 0.05, armature 0.01, frictionloss 0.2 (`g1_29dof.xml`).

## 4. Gains

`mujoco_pose_controller.py` KP/KD are identical to
`unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py` lines 18-32:

```
Kp = [60,60,60,100,40,40, 60,60,60,100,40,40, 60,40,40, 40×7, 40×7]
Kd = [1,1,1,2,1,1,        1,1,1,2,1,1,        1,1,1,   1×7,  1×7]
```

`g1_arm7_sdk_dds_example.py` (arm_sdk) uses kp = 60, kd = 1.5 for every arm
and waist joint (lines 74-75).

## 5. arm_sdk (real robot only)

Source: `unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py`.

* Topic `rt/arm_sdk`, type `unitree_hg` `LowCmd_`, published at 50 Hz
  (`control_dt_ = 0.02`). The CRC is set before every write.
* Blend weight: `motor_cmd[29].q` (`kNotUsedJoint = 29`, line 64). 1 means arm_sdk
  controls the listed joints, 0 means locomotion does. The example ramps it
  1 -> 0 over 3 s on release (line 168).
* Joints covered (`arm_joints`): 14 arm joints (15-28) **plus waist yaw, roll
  and pitch (12, 13, 14)**. Legs are never taken over, so the locomotion
  controller keeps balancing.
* The example sets `q, dq = 0, tau = 0, kp, kd` per joint. It does not set
  `mode_machine` or `motor_cmd[i].mode`.
* `ChannelFactoryInitialize(0, iface)`: the real robot uses **domain 0**, not the sim's 1.
* Waist roll and pitch are marked "INVALID for g1 23dof/29dof with waist
  locked" in `G1JointIndex`. On a waist-locked unit only waist yaw is live.
  Use `--waist yaw` on such a robot.

## 6. Geometry and frame conventions

### Robot (measured with `mj_kinematics` at q = 0, base upright)
* World and every link frame of interest (pelvis, torso_link,
  shoulder_roll/yaw, elbow, wrist links) are axis-aligned at q = 0:
  **x forward, y robot-left, z up**.
* At q = 0 the **upper arm hangs down** (shoulder_roll -> elbow unit vector
  (0.085, 0.034, -0.996)) and the **forearm points forward**
  (elbow -> wrist_roll ≈ (0.998, 0.01, -0.054)). So elbow = 0 is a 90° bend.
  **elbow = +π/2 is a straight arm**, and the range [-1.05, 2.09] means 150° of
  flexion to 30° of hyperextension.
* Joint axes at q = 0 (world): shoulder pitch (0, 0.961, 0.276) (tilted 16°),
  shoulder roll x, shoulder yaw z, elbow y, waist yaw z, waist roll x,
  waist pitch y. The waist chain is yaw -> roll -> pitch.

### MediaPipe world landmarks
Metres, origin at the hip centre, **x = image right, y = image down,
z = away from the camera** (smaller z = closer). A person facing the camera
has their left side at +x.

### Conversion used by `mediapipe_to_gmr.py`
`robot = (-z_mp, x_mp, -y_mp)`, i.e. forward = toward the camera, left = image
right, up = -y. This is a proper rotation (det = +1). For a person facing the
camera this gives the robot's anatomical left = the person's left.
`--mirror` swaps the left/right landmark indices and negates y, so the robot
behaves like a mirror image.

All targets are then expressed relative to the person's own body (pelvis and
torso frames built from the hip and shoulder lines), so the person's yaw
relative to the camera cancels out.
