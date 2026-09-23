# Running person_id

The simulated Unitree G1 copies the selected leader's arms and torso from
the camera. Pipeline:

```
camera -> MediaPipe pose (up to 4 people) -> click-selected leader
       -> one-euro filter -> visibility gate -> GMR retargeting
       -> G1 joint targets -> unitree_mujoco (rt/lowcmd)
```

One-time setup is in `../SETUP.md` (g1-env, GMR, unitree_mujoco, the pose
model). Every terminal starts with:

```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
cd ~/CITS3200/Project/person_id
```

| What | Command |
| --- | --- |
| Tests (no camera, no sim) | `python3 -m pytest tests -q` |
| Camera, no robot | `python3 leader_pose.py --dry-run` |
| Sim, robot standing (no band) | `python3 sim_standing.py` |
| Camera into the sim | `python3 leader_pose.py` (sim running) |
| Client proof video | `python3 leader_pose.py --record-demo demo.mp4` |
| Record for later | `python3 leader_pose.py --dry-run --export-landmarks exports/session.json` |
| Replay offline | `python3 leader_pose.py --replay exports/session.json --dry-run` |
| Replay into the sim | `python3 person_id_replay.py exports/session.json --report-tracking` |
| DDS check | `python3 test_dds_fixed_target.py` (sim running) |
| Xsens mocap file | `python3 xsens_replay.py recording.bvh` |

`person_id_live.py` still works and is the same as `leader_pose.py`.

---

## 1. Dry run (camera only, no DDS)

```bash
python3 leader_pose.py --dry-run
```

A window shows the camera. With more than one person, **click a person** to
make them the leader (click another person any time to switch). With
`--num-people 1` the only person is picked automatically. The overlay shows
FPS, GMR time per frame, and the status of each limb:
`live` (tracked), `hold` (just went out of view, last pose held), `ease`
(returning to neutral), `neutral`. Joint targets are printed every 15 frames.
`--dry-run` never imports `unitree_sdk2py` or opens DDS. `q`/Esc quits.

## 2. Full simulation

**Terminal 1: simulator (robot standing on the floor)**
```bash
python3 sim_standing.py
```
This runs unitree_mujoco (same scene, bridge, topics and DDS settings) with
the G1's pelvis pinned at standing height: feet on the floor, no elastic
band, no swinging. person_id doesn't balance the robot. Turning the band
off in the stock sim (`9`) makes the G1 fall: we tested holding the legs
at up to 6x the SDK gains and it still falls within seconds. Every 10 s it
prints its speed relative to real time; it should say ~1.0x.
Options: `--viewer-fps 10` (lighter), `--headless` (no window).

The stock simulator still works (`cd ~/CITS3200/Dependencies/unitree_mujoco/simulate_python
&& python3 unitree_mujoco.py`), with the robot hanging from the band. Press
`8` 4-5 times so its feet touch the floor.

**Terminal 2: person_id**
```bash
python3 leader_pose.py
```
Click the leader. The robot's arms and waist follow once a leader is
selected. Stop with `q`, Esc or Ctrl-C (or unplug the camera): the arms ease
back to where they started, then publishing stops.

Useful options (all entry points):

| Option | Effect |
| --- | --- |
| `--mirror` | Mirror mode: your left arm drives the robot's right arm. Default is anatomical (left drives left). |
| `--waist yaw` / `--waist off` | Only mimic waist yaw / keep the torso upright (use `yaw` on a waist-locked G1). |
| `--commanded-only` | Joints not mimicked (legs) are left limp (kp = kd = 0) instead of held. |
| `--legs` | Also command the legs. Sim only, band on; balance is out of scope. |
| `--max-speed 5` | Command speed cap in rad/s. |
| `--min-visibility 0.5` | MediaPipe visibility below which a limb is held, then eased to neutral. |
| `--num-people N`, `--max-match-frac 0.2`, `--leader-lost-frames 15` | Detection and leader tracking. |
| `--dds-domain 1 --dds-interface lo` | Must match `simulate_python/config.py` (these are the defaults). |

## 3. Proof video for the client

```bash
python3 leader_pose.py --record-demo demo.mp4
```
Run it with the sim open or with `--dry-run`. While you run, only the camera
frames and joint targets are stored, so the camera keeps full frame rate.
**After you quit** the video is rendered: annotated camera on the left, the
G1 (unitree_mujoco's model, driven by the same targets through the same
PD law and gains) on the right. Rendering takes about 0.1-0.3 s per frame
on the VM (no GPU), so a 1-minute take needs a few minutes. Tips: good
light, whole upper body in frame, plain clothes that contrast with the
background, move at a normal pace, hold each pose ~1 s.

A replay can be recorded the same way (stick figure instead of camera):
```bash
python3 make_synthetic_replay.py exports/synthetic.json       # test poses, no camera needed
python3 person_id_replay.py exports/synthetic.json --dry-run --record-demo exports/synthetic_demo.mp4
```

## 4. Record and replay

```bash
python3 leader_pose.py --dry-run --export-landmarks exports/session.json
python3 leader_pose.py --replay exports/session.json --dry-run          # offline
python3 person_id_replay.py exports/session.json --report-tracking      # into the sim, with numbers
```
The export format is `[{frame_id, timestamp_ms, leader_id, bbox,
landmarks_world_m: [33 x {x, y, z, visibility}]}]`. Malformed frames are
skipped with a message. `exports/` is gitignored.

## 5. Checks against the running sim

```bash
python3 test_dds_fixed_target.py
```
Sends fixed arm/waist targets over DDS, reads `rt/lowstate` back, and
passes if every joint settles within 0.05 rad, then does the same for the
return to the start pose. It exits non-zero on failure.

## 6. Xsens mocap suit (optional)

```bash
python3 xsens_replay.py ~/CITS3200/Dependencies/GMR/assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh --dry-run
python3 xsens_replay.py recording.bvh          # into the sim
```
Uses GMR's own Xsens BVH config (3ds Max-format export, cm units).

## 7. Real robot (untested on hardware)

`--real` switches the publisher to `rt/arm_sdk` (arms + waist only, Unitree's
locomotion keeps balancing the legs), ramps the arm_sdk weight 0 -> 1 over
2 s on start and back to 0 on exit, and needs the robot's network interface
and DDS domain 0:
```bash
python3 leader_pose.py --real --dds-interface enp3s0 --dds-domain 0 --waist yaw
```
Never run `leader_pose.py` without `--real` against a real robot: the default
path publishes `rt/lowcmd`, which bypasses Unitree's balance controller.
Read `INTEGRATION.md` first.

---

## Troubleshooting

**No camera / "Could not open input '0'" (VirtualBox).** The webcam has to be
passed through to the VM. Install the VirtualBox Extension Pack on the host,
then either VM window -> Devices -> Webcams -> your camera, or on the host:
`VBoxManage list webcams` and `VBoxManage controlvm "<vm name>" webcam attach .1`.
Check inside the VM with `ls /dev/video*`. If the image is black or very
slow, try `--width 640 --height 480` (MJPG is requested automatically for
cameras) and `v4l2-ctl --device=/dev/video0 --list-formats-ext`.

**"No rt/lowstate received".** The simulator isn't running, or the DDS
settings differ. `simulate_python/config.py` must have `DOMAIN_ID = 1` and
`INTERFACE = "lo"` and match `--dds-domain` / `--dds-interface`. Only
one program may publish `rt/lowcmd` at a time (see INTEGRATION.md). The
message `selected interface "lo" is not multicast-capable` is normal.

**Robot falls over / flies / swings.** Use `sim_standing.py`. In the stock
sim the band must stay on (person_id never balances). There the robot hangs
with its feet off the floor and swings when its arms move; press `8` until
the feet touch.

**"sim speed 0.3x real time" / everything lags or jerks.** The VM isn't
getting CPU time from the host. On the team VM, gnome-shell alone uses a
full core, and with 14 virtual CPUs small computations stall for 30-40 ms.
In VirtualBox (VM powered off): Settings -> System -> Processor, set the
CPU count to at most the host's *physical* core count (e.g. 4-6). Also
enable 3D acceleration under Display, and close other programs. Checks:
`sim_standing.py --headless` should report ~1.0x, and
`python3 test_dds_fixed_target.py` should pass.

**Arms go limp when person_id exits.** Intended. The stock unitree_mujoco
keeps applying the last command's torque forever, which pushes the joints
onto their limits, so the last command sent releases every joint (zero
torque). On start, every joint eases to the G1's zero posture first
(standing, arms down, forearms forward), so a slumped robot is fine.

**A limb freezes, then drops to the side.** It left the camera view or
MediaPipe's visibility for it fell below `--min-visibility`: the last pose is
held for 0.5 s, then eased to hanging over 1 s. Step back so the whole upper
body is in frame.

**Leader jumps to another person.** Click the right person again. Matching
uses position, motion and torso colour; two people in the same colour top
who cross paths can still be confused.

**Low FPS / the robot lags.** On the team VM, OpenGL is software-rendered
(llvmpipe), and the simulator's own viewer uses 3+ CPU cores. That slows
everything else down (GMR takes ~2 ms per frame alone, 30+ ms with the
viewer running). Enable 3D acceleration for the VM, give it fewer vCPUs than
the host has physical cores, or run on native Ubuntu (TESTING_SETUP.md).

**Robot mirrors instead of copying (or the reverse).** Default is
anatomical: your left arm drives the robot's left arm, so facing each other
it looks like a mirror image on screen. Add `--mirror` for mirror mode.

**Tests.** `python3 -m pytest tests -q` (about 20 s). `tests/test_mimicry.py -s`
prints the per-pose angle errors.
