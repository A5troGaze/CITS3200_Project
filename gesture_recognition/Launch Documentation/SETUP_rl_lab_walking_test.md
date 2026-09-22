# Setup: unitree_rl_lab Walking Test (`rl_lab_walking_test.py`)

This is the second walking-policy attempt, after `walking_policy_test.py`
(based on `unitree_rl_gym`) failed. This one uses `unitree_rl_lab`'s
pretrained **whole-body** (all 29 joints) velocity policy instead.

It still does **not** wire in gestures — this version's phase 3 hands
ALL 29 joints (legs + waist + arms) to the policy, so gestures can't be
layered on top yet. That's a separate architecture question the team
still needs to decide (see bottom of this doc).

## 1. What to install

### 1.1 Clone `unitree_rl_lab` (if you haven't already)

```
mkdir -p ~/CITS3200/Dependencies
cd ~/CITS3200/Dependencies
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
```

We only need the `deploy/robots/g1_29dof/` folder from this repo (the
pretrained ONNX weights + config) — we are NOT installing IsaacLab or
training anything ourselves.

### 1.2 Install onnxruntime

This policy is exported as ONNX (not `torch.jit` like the last attempt),
so it needs a different runtime:

```
pip install onnxruntime
```

### 1.3 Already required (should be installed from the last test)

- `mujoco`, `numpy`
- `unitree_sdk2py` (cloned at `~/CITS3200/Dependencies/unitree_sdk2_python`)
- `unitree_mujoco` running with our team's G1 scene

## 2. Where things point

`rl_lab_walking_test.py` expects the pretrained weights at:

```
~/CITS3200/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx
```

If your `Dependencies` folder is somewhere else, edit `POLICY_PATH` near
the top of the script.

## 3. Before running: sanity-check the ONNX shapes

Run this once, standalone, before touching the simulator:

```
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('~/CITS3200/Dependencies/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx')
for i in sess.get_inputs(): print('input:', i.name, i.shape)
for o in sess.get_outputs(): print('output:', o.name, o.shape)
"
```

**Already confirmed working**: this prints `input: obs [1, 480]` and
`output: actions [1, 29]`, matching what the script assumes. If you get
different numbers, stop and flag it — something about the observation
assembly is wrong and needs fixing before running against the robot.

## 4. Running the test

```
cd ~/Desktop/CITS3200_Project
git checkout Walking-Policy-Test
git pull
cd gesture_recognition
python rl_lab_walking_test.py
```

Same 3-phase structure as before:
1. (0-3s) ease to home pose (your fix)
2. (3-5s) hold steady
3. (5s+) hand all 29 joints to the policy, walk command ramping up
   slowly from 0 to `[0.1, 0, 0]` m/s over 5 seconds

## 5. What we already verified vs. what's still unverified

**Verified** (by reading unitree_rl_lab's own C++ deploy code —
`State_RLBase.cpp`, `observation_manager.h`, `manager_term_cfg.h`):
- The observation is 480-dim: 6 terms (ang vel, gravity, cmd, joint pos
  rel, joint vel rel, last action), each keeping its own 5-frame
  history buffer, concatenated term-block by term-block (not
  interleaved by timestep).
- `joint_ids_map` is applied on the action side as
  `motor_cmd[joint_ids_map[i]] = action[i]` — we apply the same mapping
  symmetrically on the observation side.
- ONNX input/output shapes (480 in, 29 out) match this exactly.

**NOT verified** — this can only be checked by actually running it:
- Whether the observation-side `joint_ids_map` direction is really
  symmetric with the action side. We don't have the source for
  `unitree_articulation.h` (where the real mapping might already be
  handled internally), so this is our best guess, not a certainty. If
  the robot's motion looks *systematically* wrong (not just unstable —
  e.g. joints bending the wrong way) rather than just falling over,
  check this first.
- Whether the walk is actually stable at the very small commanded
  speed (`[0.1, 0, 0]`) given the sim's slippery physics you flagged.
  Tune `TARGET_CMD` down further if it's still unstable.
- The phase 2 -> phase 3 handover: the robot's home pose might not
  exactly match `DEFAULT_JOINT_POS` (this policy's own trained
  reference pose), so there could be a small jolt right at that
  transition moment specifically.

## 6. Open architecture question (not addressed by this script)

This policy controls the whole body, so while phase 3 is running, the
arms are NOT available for gesture control. Before building further on
this, the team needs to decide: is "walk OR gesture, never both at the
same time" acceptable, or does gesture control need to work while
walking (which would mean overriding part of the policy's output,
which risks destabilising it since it wasn't trained expecting that)?

## 7. Reporting back

Please report:
- Did it walk, stumble, or fall — and specifically, did anything look
  *systematically* wrong (e.g. limbs consistently moving in the wrong
  direction) vs. just generally unstable?
- What happened right at the phase 2 -> 3 transition moment?
- A short screen recording if possible.
