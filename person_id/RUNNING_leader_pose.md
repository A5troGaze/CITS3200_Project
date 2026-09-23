# Running `leader_pose.py` — Quick Start

## 1. Activate the shared environment

```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```

Your terminal prompt should now start with `(g1-env)`.

## 2. Confirm the pose model file is present

```bash
ls -la ~/CITS3200/Dependencies/Models/pose_landmarker.task
```

Should be several MB, not 0 bytes. If yours lives elsewhere, pass
`--model /path/to/pose_landmarker.task` or set `CITS3200_MODELS_DIR`.

## 3. Navigate to the script

```bash
cd ~/CITS3200/Project/person_id
```

## 4. Pick a mode

### Camera, tracking only (no robot)

```bash
python3 leader_pose.py
```

A window opens showing the webcam feed. With more than one person in
frame, click one to select them as the leader — click again anytime to
switch. With `--num-people 1` (single person expected) the sole detection
is selected automatically, no click needed. Press `q` to quit.

```bash
# Record a session for later replay/testing:
python3 leader_pose.py --export-landmarks session.json
```

### Replay a recording (no camera, no robot)

Feeds a previously recorded `--export-landmarks` file through the
retargeter and prints the 29 joint targets per frame — useful for
checking the retargeting math or testing in the VM with no webcam
attached at all.

```bash
python3 leader_pose.py --replay session.json --dry-run
```

### Full sim mode (camera -> retargeting -> MuJoCo)

Needs unitree_mujoco running first (see `SETUP.md`'s "Person ID / MuJoCo
bridge" section for one-time setup). Three-terminal sequence:

**Terminal 1 — the simulator:**
```bash
cd ~/CITS3200/Dependencies/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
```
Wait for the MuJoCo window to open with the G1 standing, then press `9` to
release the elastic band holding it up (see `SETUP.md` for what the other
band keys do). The pipeline only commands the arms — the band is what
keeps the rest of the robot from falling over.

**Terminal 2 — this pipeline:**
```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
cd ~/CITS3200/Project/person_id
python3 leader_pose.py --mujoco --dds-domain 1 --dds-interface lo
```
(`--dds-domain 1 --dds-interface lo` are already the defaults — shown
explicitly here since they must match `simulate_python/config.py`'s
`DOMAIN_ID`/`INTERFACE`, not because you need to pass them.)

Click a person in the preview window to select them as leader (or use
`--num-people 1` to skip the click). Their tracked arm movements should
now mirror onto the simulated G1's arms.

**Terminal 3 (optional) — webcam on Windows, MuJoCo in WSL:**
If the webcam can't be reached from inside WSL, run the capture on
Windows and forward landmarks over UDP instead of running `leader_pose.py`
directly in Terminal 2:
```powershell
# Windows terminal, in a Python env with mediapipe + opencv-python installed:
python live_pose_sender.py --host <WSL IP address>
```
```bash
# WSL terminal, instead of Terminal 2 above:
python3 live_pose_receiver.py --dds-domain 1 --dds-interface lo
```
Note: `live_pose_receiver.py` calls `MujocoLink.update_pose()`, which only
stores raw landmarks — it does not retarget or publish `LowCmd`. Retargeting
is only wired up through `leader_pose.py --mujoco`'s `publish_target()`
path; extending the receiver to retarget is future work, not done here.

### `--dry-run` without `--mujoco`

`--dry-run` never opens a DDS channel and never imports `unitree_sdk2py`,
so `leader_pose.py --dry-run` (camera or `--replay`) works even without
unitree_mujoco/unitree_sdk2py installed at all — useful for iterating on
the retargeting math alone.

## Troubleshooting

- **Camera window doesn't open / wrong device:** machine-specific camera
  setup, not this script — check with the team.
- **`mujoco_link.KP/KD are not filled in`**: expected until someone copies
  the exact `Kp`/`Kd` arrays from
  `unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py` into
  `person_id/mujoco_link.py` — see that file's module docstring and this
  task's final summary for why they were left blank rather than guessed.
- **Robot falls over in the sim:** the elastic band (config
  `ENABLE_ELASTIC_BAND` / in-sim key `9`) isn't enabled — this pipeline
  never balances or walks, only mirrors arms.

## When you're done

```bash
deactivate
```
