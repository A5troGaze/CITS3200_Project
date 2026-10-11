"""Publisher logic without DDS: the sim controller's per-tick command, the
arm_sdk weight ramp, and the guarantee that --dry-run never imports
unitree_sdk2py."""

import subprocess
import sys
import os

import numpy as np
import pytest

from arm_sdk_publisher import ARM_ONLY_JOINTS, ARM_SDK_JOINTS, ArmSdkPublisher
from command_shaping import CommandShaper
from g1_gains import KD, KP
from mujoco_pose_controller import MujocoPoseController

PERSON_ID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _controller(**kwargs):
    """A controller as _on_low_state leaves it after the first LowState."""
    ctl = MujocoPoseController(gravity_comp=False, **kwargs)
    start = np.linspace(-0.2, 0.2, 29)
    ctl.latest_q = start.copy()
    ctl.hold_positions = np.zeros(29) if ctl.home == "zero" else start.copy()
    ctl.shaper = CommandShaper(start, ctl.control_dt, engage_s=0.0)
    ctl.shaper.target = ctl.hold_positions.copy()
    ctl.ready.set()
    return ctl, start


def test_hold_mode_holds_uncommanded_joints_with_sdk_gains():
    ctl, start = _controller(home="current")
    ctl.set_targets({18: 1.0})
    rows = ctl.build_command()
    for i, (q, kp, kd, tau) in enumerate(rows):
        assert (kp, kd) == (KP[i], KD[i])
        if i != 18:
            assert q == pytest.approx(start[i])
    assert rows[18][0] > start[18]  # moving toward the target, rate-limited


def test_commanded_only_leaves_other_joints_limp():
    ctl, start = _controller(commanded_only=True)
    ctl.set_targets({18: 1.0, 25: 1.0})
    rows = ctl.build_command()
    for i, (q, kp, kd, tau) in enumerate(rows):
        if i in (18, 25):
            assert kp > 0 and kd > 0
        else:
            assert (kp, kd, tau) == (0.0, 0.0, 0.0)


def test_legs_rejected_unless_enabled():
    ctl, _ = _controller()
    with pytest.raises(ValueError):
        ctl.set_targets({3: 0.5})
    ctl_legs, _ = _controller(allow_legs=True)
    ctl_legs.set_targets({3: 0.5})


def test_non_finite_target_rejected():
    ctl, _ = _controller()
    with pytest.raises(ValueError):
        ctl.set_targets({18: float("nan")})


def test_gravity_feed_forward_on_upper_body_not_legs():
    ctl = MujocoPoseController(gravity_comp=True)
    start = np.zeros(29)
    ctl.hold_positions, ctl.latest_q = start.copy(), start.copy()
    ctl.shaper = CommandShaper(start, ctl.control_dt, engage_s=0.0)
    ctl.set_targets({15: -1.57})     # left shoulder pitch: arm forward, loaded by gravity
    ctl.latest_q[15] = -1.57
    rows = ctl.build_command()
    assert abs(rows[15][3]) > 0.5    # N*m, holding the arm up
    assert all(rows[i][3] == 0.0 for i in range(12))   # legs: no feed-forward


def test_home_zero_eases_every_joint_to_zero_posture():
    ctl, start = _controller()
    first = ctl.build_command()
    assert all(abs(q - s) < 0.02 for (q, *_), s in zip(first, start))   # no jump
    for _ in range(1000):
        rows = ctl.build_command()
    assert all(abs(q) < 1e-3 for q, *_ in rows)


def test_return_to_start_goes_back():
    ctl, start = _controller(home="current")
    ctl.set_targets({18: 1.0})
    for _ in range(500):
        ctl.build_command()
    ctl.clear_targets()
    for _ in range(1000):
        ctl.build_command()
    assert ctl.shaper.q[18] == pytest.approx(start[18], abs=0.02)


def test_arm_sdk_weight_ramps_up_and_down_without_jumps():
    pub = ArmSdkPublisher(real=False, control_hz=50.0, ramp_s=2.0, log=lambda *a: None)
    pub.init(start_q=np.zeros(29))
    pub.engage()
    weights = [pub.build_message()[0] for _ in range(150)]
    assert weights[0] > 0 and weights[-1] == 1.0
    assert max(np.diff(weights)) <= pub.dt / pub.ramp_s + 1e-12
    pub.weight_target = 0.0
    down = [pub.build_message()[0] for _ in range(150)]
    assert down[-1] == 0.0
    assert min(np.diff(down)) >= -pub.dt / pub.ramp_s - 1e-12


def test_arm_sdk_covers_only_arms_by_default():
    assert ARM_ONLY_JOINTS == list(range(15, 29))
    pub = ArmSdkPublisher(real=False, log=lambda *a: None)
    pub.init(start_q=np.zeros(29))
    for joint in (0, 12):           # a leg, the waist: left to the balance controller
        with pytest.raises(ValueError):
            pub.set_targets({joint: 0.1})
    _, q = pub.build_message()
    assert set(q) == set(ARM_ONLY_JOINTS)
    with_waist = ArmSdkPublisher(real=False, log=lambda *a: None, joints=ARM_SDK_JOINTS)
    with_waist.init(start_q=np.zeros(29))
    assert set(with_waist.build_message()[1]) == set(range(12, 29))


def test_arm_sdk_real_needs_an_interface():
    with pytest.raises(ValueError):
        ArmSdkPublisher(real=True, interface=None)


def test_dry_run_never_imports_unitree_sdk():
    code = (
        "import sys, types; sys.argv=['x'];"
        "import leader_pose, person_id_replay, mimic_pipeline, arm_sdk_publisher, mujoco_pose_controller;"
        "args = leader_pose.parse_args(['--dry-run']);"
        "assert person_id_replay.build_publisher(args) is None;"
        "print(any(m.startswith('unitree_sdk2py') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=PERSON_ID, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_lost_leader_holds_then_eases_to_neutral(pipeline):
    from synthetic_poses import base_poses

    pipeline.reset()
    t_pose = [p for p in base_poses() if p.name == "t_pose"][0]
    t = 0.0
    for _ in range(5):
        pipeline.step(t_pose.landmarks, t)
        t += 1 / 30
    statuses = []
    while t < 2.5:
        r = pipeline.step(None, t)
        statuses.append(r.status["left_upper"])
        t += 1 / 30
    assert statuses[0] == "hold" and "ease" in statuses and statuses[-1] == "neutral"
    assert r.info["segments"].upper["left"][2] < -0.99
