## G1 joint DOF / angle / velocity limit analysis

Scope note first: your connected OneDrive folder (`CITS3200_Project`, GitHub repo `A5troGaze/CITS3200_Project`) only contains Documentation plus two subsystems, checked across every branch (`main`, `person_id`, `personid-multiple`, `Gesture`, `Recognize-Gestures`, `Formal-Gesture-Testing`, `native-position-test`, `Documentation`):

- `person_id/` — MediaPipe camera pose detection, leader tracking, landmark export (no robot/joint code)
- `gesture_recognition/` (on the `Gesture` branches) — classifies discrete gestures from landmarks into labels like "thumbs up" (no robot/joint code)

There is no kinematics, retargeting, or robot-joint-command code checked into this repo on any branch. Per `SETUP.md`, that layer runs on `unitree_sdk2_python` and `GMR` (YanjieZe/GMR), both cloned into `~/CITS3200/Dependencies/` on your dev machine and explicitly excluded from the repo ("Never commit anything from `~/CITS3200/Dependencies/` to Git"). So there was nothing in your own code to check DOF/limits against — the analysis below is against those two upstream libraries directly, pulled fresh from their GitHub repos, since that's where this logic actually lives (or is supposed to).

### 1. DOF count: matches, conditionally

Your stack targets the Unitree G1 **29-DOF** variant (no hands): `GMR`'s robot asset table (`general_motion_retargeting/params.py`) maps `"unitree_g1"` to `assets/unitree_g1/g1_mocap_29dof.xml`, which defines the same 29 joints, in the same order, as `G1JointIndex` in `unitree_sdk2_python`'s low-level example (`G1_NUM_MOTOR = 29`). 12 leg + 3 waist + 14 arm = 29. Indices line up 1:1 (`LeftHipPitch = 0` ... `RightWristYaw = 28`).

This match only holds as long as everyone uses the same asset consistently. Two ways it silently breaks:

- Someone switches to `g1_23dof_*` (locked/absent waist roll+pitch, 5-DOF arms, no wrist pitch/yaw) or `g1_29dof_with_hand_*` (29 + 7×2 hand DOF) for one part of the pipeline but not another — e.g. simulate in MuJoCo with 23-DOF but send commands assuming 29.
- Your **actual physical unit** isn't the 29-DOF-no-hands variant. This can't be confirmed from code; check it in-app: **Device → Data → Robot → Machine Type**, and cross-reference against the table in `unitree_ros/robots/g1_description/README.md`.

`g1_joint_limits.py` (attached) includes `check_dof_count()` for exactly this — feed it whatever your pipeline treats as "the robot's joints" at each stage and it flags a mismatch instead of letting it fail silently downstream.

### 2. Joint angle limits: already enforced correctly, nothing to fix

GMR always applies `mink.ConfigurationLimit(self.model)` on every IK solve (`general_motion_retargeting/motion_retarget.py`, unconditional, not behind any flag) — this reads the `<joint range="...">` values baked into `g1_mocap_29dof.xml` and makes them a hard constraint, so IK output cannot exceed those ranges.

I compared every one of those 29 ranges against Unitree's own hardware spec (`g1_29dof_rev_1_0.urdf`, official `unitree_ros` repo) and GMR's ranges are a safe subset on every joint — either identical (knee, ankle pitch/roll, waist roll/pitch, all three wrist joints) or deliberately tighter (hip pitch/roll/yaw, waist yaw, shoulder pitch/roll/yaw, elbow). So position-limit-wise, retargeted motion built on this default asset already respects the real robot's angle limits, with margin on several joints. No change needed here — just don't hand-edit those MJCF ranges wider without re-checking against the hardware table below.

### 3. Joint velocity limits: not matched to the robot, and off by default

This is the actual gap.

- `GeneralMotionRetargeting.__init__` has `use_velocity_limit: bool = False` as its default (current source, `general_motion_retargeting/motion_retarget.py` line 21). If your retargeting call site doesn't explicitly pass `use_velocity_limit=True`, **IK runs with no velocity constraint at all.** Worth flagging on its own: GMR's README changelog claims this defaults to `True` ("2025-08-24: ... `use_velocity_limit=True` by default") — the installed source contradicts its own docs. Don't trust the README on this; check whichever commit actually got `pip install -e .`'d into `g1-env`.
- Even with it enabled, the limit applied is one flat value, `3*np.pi` (~9.42 rad/s), for all 29 motors:
  ```python
  VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
  self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))
  ```
  That's not a safety bug by itself — 9.42 rad/s is below every joint's real max in the table below, so it can't ask a motor to exceed its rated speed. But it isn't "matching the model's limitations" either: it caps arm joints (rated 37 rad/s) at a quarter of their real speed while treating them the same as hip/knee joints (rated 20-32 rad/s), which is arbitrary rather than derived from the robot.

Fix is mechanical: build the `VELOCITY_LIMITS` dict from real per-joint numbers instead of the flat constant, and make sure `use_velocity_limit=True` is actually passed at the call site. `g1_joint_limits.py` has the exact per-joint values and the two-line patch is in a comment at the bottom of that file.

One more thing worth being explicit about: IK-time velocity limiting only constrains the *solve*. Whatever turns retargeted frames into the actual `LowCmd` stream sent over `unitree_sdk2py` (frame-rate conversion, interpolation, any smoothing, teleop network jitter) sits downstream of IK and isn't covered by `mink.VelocityLimit` at all. If that assembly step can introduce a large position jump between consecutive commands, effective velocity can exceed the IK-time bound even with everything above fixed. `g1_joint_limits.py` includes `validate_command()` for exactly that — run it as a last check on the per-tick command dict right before it goes out, not just at IK time.

### 4. Reference: full 29-DOF limit table (source: Unitree `g1_29dof_rev_1_0.urdf`)

| # | Joint | Lower (rad) | Upper (rad) | Lower (deg) | Upper (deg) | Max velocity (rad/s) | Max effort (N·m) |
|---|---|---|---|---|---|---|---|
| 0 | left_hip_pitch | -2.5307 | 2.8798 | -145.0 | 165.0 | 32 | 88 |
| 1 | left_hip_roll | -0.5236 | 2.9671 | -30.0 | 170.0 | 20 | 139 |
| 2 | left_hip_yaw | -2.7576 | 2.7576 | -158.0 | 158.0 | 32 | 88 |
| 3 | left_knee | -0.0873 | 2.8798 | -5.0 | 165.0 | 20 | 139 |
| 4 | left_ankle_pitch | -0.8727 | 0.5236 | -50.0 | 30.0 | 30 | 35 |
| 5 | left_ankle_roll | -0.2618 | 0.2618 | -15.0 | 15.0 | 30 | 35 |
| 6 | right_hip_pitch | -2.5307 | 2.8798 | -145.0 | 165.0 | 32 | 88 |
| 7 | right_hip_roll | -2.9671 | 0.5236 | -170.0 | 30.0 | 20 | 139 |
| 8 | right_hip_yaw | -2.7576 | 2.7576 | -158.0 | 158.0 | 32 | 88 |
| 9 | right_knee | -0.0873 | 2.8798 | -5.0 | 165.0 | 20 | 139 |
| 10 | right_ankle_pitch | -0.8727 | 0.5236 | -50.0 | 30.0 | 30 | 35 |
| 11 | right_ankle_roll | -0.2618 | 0.2618 | -15.0 | 15.0 | 30 | 35 |
| 12 | waist_yaw | -2.618 | 2.618 | -150.0 | 150.0 | 32 | 88 |
| 13 | waist_roll | -0.52 | 0.52 | -29.8 | 29.8 | 30 | 35 |
| 14 | waist_pitch | -0.52 | 0.52 | -29.8 | 29.8 | 30 | 35 |
| 15 | left_shoulder_pitch | -3.0892 | 2.6704 | -177.0 | 153.0 | 37 | 25 |
| 16 | left_shoulder_roll | -1.5882 | 2.2515 | -91.0 | 129.0 | 37 | 25 |
| 17 | left_shoulder_yaw | -2.618 | 2.618 | -150.0 | 150.0 | 37 | 25 |
| 18 | left_elbow | -1.0472 | 2.0944 | -60.0 | 120.0 | 37 | 25 |
| 19 | left_wrist_roll | -1.9722 | 1.9722 | -113.0 | 113.0 | 37 | 25 |
| 20 | left_wrist_pitch | -1.6144 | 1.6144 | -92.5 | 92.5 | 22 | 5 |
| 21 | left_wrist_yaw | -1.6144 | 1.6144 | -92.5 | 92.5 | 22 | 5 |
| 22 | right_shoulder_pitch | -3.0892 | 2.6704 | -177.0 | 153.0 | 37 | 25 |
| 23 | right_shoulder_roll | -2.2515 | 1.5882 | -129.0 | 91.0 | 37 | 25 |
| 24 | right_shoulder_yaw | -2.618 | 2.618 | -150.0 | 150.0 | 37 | 25 |
| 25 | right_elbow | -1.0472 | 2.0944 | -60.0 | 120.0 | 37 | 25 |
| 26 | right_wrist_roll | -1.9722 | 1.9722 | -113.0 | 113.0 | 37 | 25 |
| 27 | right_wrist_pitch | -1.6144 | 1.6144 | -92.5 | 92.5 | 22 | 5 |
| 28 | right_wrist_yaw | -1.6144 | 1.6144 | -92.5 | 92.5 | 22 | 5 |

This table (as literal Python constants, plus the validation/clamp helper functions referenced above) is in the attached `g1_joint_limits.py`.

### 5. What this doesn't cover

I don't have visibility into whichever branch or local files (if any) already hold retargeting/robot-control code your kinematics subgroup may be writing right now on the dev machine — that lives outside this repo and outside this OneDrive folder by design. If that code exists somewhere reachable (a branch not yet pushed, a private repo, a folder you can share), point me at it and I'll check it directly against this table instead of against upstream GMR's defaults. If your physical G1 isn't the 29-DOF-no-hands variant, say which one and I'll regenerate the table from the matching URDF.

### Sources

- [unitree_ros — robots/g1_description (23/29-DOF variant table, MJCF/URDF)](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description)
- [unitree_ros — g1_29dof_rev_1_0.urdf (joint limit source used above)](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.urdf)
- [unitree_sdk2_python — example/g1/low_level/g1_low_level_example.py (G1JointIndex, G1_NUM_MOTOR)](https://github.com/unitreerobotics/unitree_sdk2_python/blob/main/example/g1/low_level/g1_low_level_example.py)
- [YanjieZe/GMR — general_motion_retargeting/motion_retarget.py (IK limits, use_velocity_limit)](https://github.com/YanjieZe/GMR/blob/main/general_motion_retargeting/motion_retarget.py)
- [YanjieZe/GMR — general_motion_retargeting/params.py (robot asset mapping)](https://github.com/YanjieZe/GMR/blob/main/general_motion_retargeting/params.py)
- [YanjieZe/GMR — assets/unitree_g1/g1_mocap_29dof.xml (retargeting MJCF joint ranges)](https://github.com/YanjieZe/GMR/blob/main/assets/unitree_g1/g1_mocap_29dof.xml)
- [YanjieZe/GMR — README (velocity-limit changelog claim)](https://github.com/YanjieZe/GMR/blob/main/README.md)
