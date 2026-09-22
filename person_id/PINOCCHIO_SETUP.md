# Pinocchio Retargeting Setup

This adds a new retargeting method that uses Pinocchio (inverse kinematics
against the real G1 skeleton) instead of the geometric math currently used
in `pose_retargeting.py`. It should give more accurate joint angles since
it accounts for the robot's actual arm proportions instead of estimating
from human ones.

## 1. Install Pinocchio

In the shared venv:

```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
pip install pin --break-system-packages
```

Check it installed correctly:

```bash
python3 -c "import pinocchio; print(pinocchio.__version__)"
```

## 2. Find the G1 URDF file

We don't know the exact file path on your machine, so search for it first:

```bash
find ~/CITS3200 -iname "*g1*urdf*" 2>/dev/null
find ~/CITS3200 -iname "*.urdf" 2>/dev/null
```

If nothing shows up, grab the official one from Unitree's repo:

```bash
mkdir -p ~/CITS3200/Dependencies/g1_description
curl -o ~/CITS3200/Dependencies/g1_description/g1_29dof_rev_1_0.urdf \
  https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/g1_29dof_rev_1_0.urdf
```

Then open `pinocchio_retargeting.py` and set the path near the top:

```python
G1_URDF_PATH = "PUT_THE_REAL_PATH_HERE"
```

## 3. Find the wrist frame names

Run the file on its own. It'll load the model and print every frame name
in it, plus a shortlist of likely wrist matches:

```bash
python3 pinocchio_retargeting.py
```

Check the "Likely wrist candidates" section of the output, then set the
left/right names near the top of the file:

```python
LEFT_WRIST_FRAME = "..."
RIGHT_WRIST_FRAME = "..."
```

Run `python3 pinocchio_retargeting.py` again. If it prints the DOF/frame
count with no errors, both values are correct.

## 4. Point person_id_live.py at the new retargeting

Find this import near the top of `person_id_live.py`:

```python
from pose_retargeting import mediapipe_landmarks_to_xyz, retarget_arms_indexed
```

Change it to:

```python
from pose_retargeting import mediapipe_landmarks_to_xyz   # keep this import
from pinocchio_retargeting import load_model, retarget_arms_pinocchio

# run once at startup, before the main loop:
pin_model, pin_data = load_model()
_q_prev = None
```

Then find where the script calls:

```python
joint_targets = retarget_arms_indexed(xyz)
```

and replace it with:

```python
joint_targets, _q_prev = retarget_arms_pinocchio(xyz, pin_model, pin_data, q_prev=_q_prev)
```

`joint_targets` comes back in the same `{motor_index: angle_rad}` format
as before, so nothing else in the script (including
`mujoco_pose_controller.set_targets(...)`) needs to change.

## 5. Run it

Same as usual, two terminals:

```bash
cd ~/CITS3200/Dependencies/unitree_mujoco/simulate_python
python unitree_mujoco.py
```

```bash
cd ~/CITS3200/Project/person_id   # or wherever the file actually lives
python3 person_id_live.py --input <camera index or video file>
```

## Troubleshooting

- **FileNotFoundError on load**: the URDF path from step 2 is wrong.
- **KeyError / frame not found**: the wrist frame names from step 3 are wrong.
- **Arms move but look off or jerky**: that's a real tuning issue, not a
  setup mistake — let Lithi know with a screen recording if possible so we
  can debug it properly rather than guessing.
