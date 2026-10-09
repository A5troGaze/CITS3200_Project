"""
Definition of "the robot copies the leader", tested headless (no camera,
no DDS): synthetic MediaPipe poses -> the full pipeline (filter, adapter,
gate, GMR, name->motor mapping, clamp) -> unitree_mujoco's G1 MJCF.

For every pose, the robot's upper-arm and forearm directions (from MuJoCo
body positions, in the robot torso frame) must be within 15 deg of the
person's (in the person's torso frame). The default pipeline mimics the
arms only (the torso is left to the balance controller). Torso mimicry
(--waist 3dof) is checked separately: the torso's up and lateral axes
relative to the pelvis within 15 deg of the person's.

The reference is the nearest direction the G1 can physically take:
* joint limits: a pose past a limit must leave that joint at the limit
  (test_unreachable_*),
* self-collision: a hanging arm is kept HIP_CLEARANCE_DEG outward so the
  hand clears the G1's hip links (mediapipe_to_gmr.hip_clearance), so the
  reference for a hanging arm is the person's direction after that rule.
The raw person-vs-robot error is printed for every pose (pytest -s).

Kinematic: joint targets written straight into qpos.
Dynamic: the fixed-base G1 stepped with the simulator's PD law, the SDK
gains and gravity feed-forward (g1_sim.PdSim), 3.5 s to settle.
"""

import math

import numpy as np
import pytest

from g1_sim import PdSim, angle_deg, load_sim_model, segment_vectors, set_joint_positions
from mediapipe_to_gmr import hip_clearance
from synthetic_poses import all_poses, build_pose, rot_y, unit

TOL_DEG = 15.0
POSES = all_poses()


@pytest.fixture(scope="module")
def sim():
    import mujoco

    model = load_sim_model(fixed_base=True)
    return model, mujoco.MjData(model)


def run_static(pipeline, landmarks, frames=3):
    pipeline.reset()
    result = None
    for i in range(frames):
        result = pipeline.step(landmarks, i / 30.0)
    return result


def reference_dirs(pose):
    """Nearest physically feasible directions for the pose (person frame)."""
    upper, fore = {}, {}
    for side in ("left", "right"):
        upper[side], fore[side] = hip_clearance(side, pose.upper[side], pose.fore[side])
    return upper, fore


def segment_errors(pose, upper, fore, torso_rel, reference=True, torso=False):
    ref_u, ref_f = reference_dirs(pose) if reference else (pose.upper, pose.fore)
    errs = {}
    for side in ("left", "right"):
        errs[f"{side}_upper"] = angle_deg(upper[side], ref_u[side])
        errs[f"{side}_fore"] = angle_deg(fore[side], ref_f[side])
    if torso:
        errs["torso_up"] = angle_deg(torso_rel[:, 2], pose.torso[:, 2])
        errs["torso_lateral"] = angle_deg(torso_rel[:, 1], pose.torso[:, 1])
    return errs


@pytest.fixture(scope="module")
def torso_pipeline():
    from gmr_retarget import GmrRetargeter
    from mimic_pipeline import MimicPipeline

    return MimicPipeline(waist="3dof", retargeter=GmrRetargeter(budget_ms=1000.0))


TORSO_POSES = [p for p in POSES if p.name.startswith(("lean", "torso"))]


@pytest.mark.parametrize("pose", TORSO_POSES, ids=[p.name for p in TORSO_POSES])
def test_torso_mimicry_when_waist_enabled(torso_pipeline, sim, pose):
    model, data = sim
    result = run_static(torso_pipeline, pose.landmarks)
    set_joint_positions(model, data, result.q)
    errs = segment_errors(pose, *segment_vectors(model, data), torso=True)
    bad = {k: round(v, 1) for k, v in errs.items() if v > TOL_DEG}
    assert not bad, f"{pose.name}: segments over {TOL_DEG} deg: {bad}"


def _report(kind, pose, errs, raw):
    worst = max(errs, key=errs.get)
    print(f"{kind:4s} {pose.name:34s} max {errs[worst]:5.1f} deg ({worst}); "
          f"raw vs person max {max(raw.values()):5.1f}")


@pytest.mark.parametrize("pose", POSES, ids=[p.name for p in POSES])
def test_kinematic_mimicry(pipeline, sim, pose):
    model, data = sim
    result = run_static(pipeline, pose.landmarks)
    set_joint_positions(model, data, result.q)
    vecs = segment_vectors(model, data)
    errs = segment_errors(pose, *vecs)
    _report("kin", pose, errs, segment_errors(pose, *vecs, reference=False))
    bad = {k: round(v, 1) for k, v in errs.items() if v > TOL_DEG}
    assert not bad, f"{pose.name}: segments over {TOL_DEG} deg: {bad}"


@pytest.mark.parametrize("pose", POSES, ids=[p.name for p in POSES])
def test_dynamic_mimicry(pipeline, pose):
    result = run_static(pipeline, pose.landmarks)
    sim = PdSim()
    sim.set_targets(result.targets)
    sim.step(int(3.5 / sim.model.opt.timestep))
    vecs = segment_vectors(sim.model, sim.data)
    errs = segment_errors(pose, *vecs)
    _report("dyn", pose, errs, segment_errors(pose, *vecs, reference=False))
    bad = {k: round(v, 1) for k, v in errs.items() if v > TOL_DEG}
    assert not bad, f"{pose.name}: segments over {TOL_DEG} deg: {bad}"
    assert np.all(np.isfinite(sim.data.qpos))


def test_unreachable_arm_behind_goes_to_shoulder_pitch_limit(pipeline):
    # Upper arm straight back, horizontal: past the G1's shoulder pitch range.
    pose = build_pose("arm_back", (-1, 0, -0.05), (-1, 0, -0.05), (0, 0, -1), (0, 0, -1))
    result = run_static(pipeline, pose.landmarks, frames=5)
    lo, hi = pipeline.gmr.model.joint("left_shoulder_pitch_joint").range
    assert result.q["left_shoulder_pitch_joint"] == pytest.approx(hi, abs=0.05)


def test_unreachable_deep_lean_goes_to_waist_pitch_limit(torso_pipeline):
    pose = build_pose("deep_lean", rot_y(50).T @ unit(0, 0, -1), rot_y(50).T @ unit(0, 0, -1), torso=rot_y(50))
    result = run_static(torso_pipeline, pose.landmarks)
    lo, hi = torso_pipeline.gmr.model.joint("waist_pitch_joint").range
    assert result.q["waist_pitch_joint"] == pytest.approx(hi, abs=0.02)


def test_all_targets_within_real_joint_limits(pipeline):
    from g1_joint_limits import validate_command

    for pose in POSES:
        result = run_static(pipeline, pose.landmarks)
        assert validate_command(result.q) == [], pose.name


def test_only_arms_commanded_by_default(pipeline):
    from gmr_retarget import LEG_JOINTS
    from g1_joint_limits import G1_29DOF_JOINT_LIMITS

    result = run_static(pipeline, POSES[0].landmarks)
    leg_idx = {G1_29DOF_JOINT_LIMITS[n].index for n in LEG_JOINTS}
    assert not leg_idx & set(result.targets)
    assert set(result.targets) == set(range(15, 29))   # arms only: no legs, no waist
    assert abs(result.q["waist_pitch_joint"]) < 0.05 and abs(result.q["waist_yaw_joint"]) < 0.05


def _robot_arm_up(result, sim, side):
    model, data = sim
    set_joint_positions(model, data, result.q)
    upper, _, _ = segment_vectors(model, data)
    return upper[side][2] > 0.9


def test_left_right_not_swapped(pipeline, sim):
    pose = [p for p in POSES if p.name == "left_up_right_down"][0]
    result = run_static(pipeline, pose.landmarks)
    assert _robot_arm_up(result, sim, "left")
    assert not _robot_arm_up(result, sim, "right")


def test_mirror_swaps_sides(sim):
    from mimic_pipeline import MimicPipeline

    mirrored = MimicPipeline(mirror=True)
    pose = [p for p in POSES if p.name == "left_up_right_down"][0]
    result = run_static(mirrored, pose.landmarks)
    assert _robot_arm_up(result, sim, "right")
    assert not _robot_arm_up(result, sim, "left")


def test_mirror_reflects_direction(sim):
    """Mirror mode: the robot's right arm reproduces the person's left arm
    reflected across the sagittal plane (forward stays forward)."""
    from mimic_pipeline import MimicPipeline

    model, data = sim
    pose = build_pose("left_forward_out", unit(1, 1, 0), unit(1, 1, 0), (0, 0, -1), (0, 0, -1))
    result = run_static(MimicPipeline(mirror=True), pose.landmarks)
    set_joint_positions(model, data, result.q)
    upper, _, _ = segment_vectors(model, data)
    expected = pose.upper["left"] * np.array([1, -1, 1])
    assert angle_deg(upper["right"], expected) < TOL_DEG


def test_same_body_turned_30_degrees_gives_same_joints(pipeline):
    by_name = {p.name: p for p in POSES}
    for name in ("elbows_90_forearms_forward", "goalpost", "wave", "torso_twisted"):
        a = run_static(pipeline, by_name[name].landmarks).q
        b = run_static(pipeline, by_name[name + "_rot30"].landmarks).q
        for joint in a:
            assert abs(a[joint] - b[joint]) < math.radians(2), (name, joint)


def test_latency_budget():
    """GMR must fit in one 30 fps frame (33 ms) while tracking a continuous
    motion, and so must the whole retarget step (filter + adapter + GMR).

    Asserted on the wall-clock median and on the CPU-time p95. Wall-clock
    tails on the team VM are dominated by the hypervisor: 20 plain 40x40
    numpy solves have a 0.8 ms median there but stall up to ~45 ms. Wall
    p95/max are printed so they are still visible."""
    from mimic_pipeline import MimicPipeline
    from synthetic_poses import base_poses, blend

    seq = base_poses()
    pipeline = MimicPipeline()   # default (real) GMR time budget
    rows = []
    t = 0.0
    for a, b in zip(seq, seq[1:]):
        for k in range(15):  # 0.5 s per transition at 30 fps
            s = k / 15.0
            r = pipeline.step(blend(a, b, s).landmarks, t)
            t += 1 / 30.0
            ms = r.timings_ms
            rows.append((ms["gmr"], ms["total"], ms["gmr_cpu"], ms["total_cpu"]))
    gmr, total, gmr_cpu, total_cpu = np.array(rows).T
    stat = lambda x: f"median {np.median(x):.2f} / p95 {np.percentile(x, 95):.2f} / max {x.max():.2f} ms"
    print(f"\n{len(rows)} frames. GMR wall {stat(gmr)}; GMR cpu {stat(gmr_cpu)}")
    print(f"whole retarget step wall {stat(total)}; cpu {stat(total_cpu)}")
    assert np.median(gmr) < 33.0 and np.median(total) < 33.0
    assert np.percentile(gmr_cpu, 95) < 33.0 and np.percentile(total_cpu, 95) < 33.0
