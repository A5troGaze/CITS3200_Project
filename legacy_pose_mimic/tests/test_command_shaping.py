"""Controller command shaping (replaces test_mujoco_link_interp.py, whose
module mujoco_link.py was merged into mujoco_pose_controller.py)."""

import numpy as np

from command_shaping import CommandShaper, step_toward
from g1_joint_limits import G1_29DOF_JOINT_LIMITS

PUBLISH_HZ = 500.0
TARGET_HZ = 30.0


def test_interpolated_steps_never_exceed_velocity_limit_per_tick():
    rng = np.random.default_rng(0)
    velocity_limits = np.array([lim.velocity for lim in G1_29DOF_JOINT_LIMITS.values()])
    lowers = np.array([lim.lower for lim in G1_29DOF_JOINT_LIMITS.values()])
    uppers = np.array([lim.upper for lim in G1_29DOF_JOINT_LIMITS.values()])

    dt = 1.0 / PUBLISH_HZ
    max_step = velocity_limits * dt
    published = np.zeros(29)
    target = published.copy()
    ticks_per_target = int(round(PUBLISH_HZ / TARGET_HZ))
    for tick in range(500):
        if tick % ticks_per_target == 0:
            target = rng.uniform(lowers, uppers)
        new_published = step_toward(published, target, max_step)
        assert np.all(np.abs(new_published - published) <= max_step + 1e-9)
        published = new_published


def test_shaper_follows_closely_without_overshoot():
    """A 1 rad target change reaches 95 % within 0.25 s and never overshoots."""
    dt = 1.0 / PUBLISH_HZ
    s = CommandShaper(np.zeros(29), dt, engage_s=0.0)
    s.set_target(18, 1.0)
    q, t95 = [], None
    for k in range(int(1.0 / dt)):
        v = s.tick()[18]
        q.append(v)
        if t95 is None and v >= 0.95:
            t95 = (k + 1) * dt
    assert max(q) <= 1.0 + 1e-12
    assert t95 is not None and t95 < 0.25, t95


def test_shaper_removes_30hz_staircase():
    """Vision targets arrive at 30 Hz; the 500 Hz command must not step."""
    dt = 1.0 / PUBLISH_HZ
    s = CommandShaper(np.zeros(29), dt, engage_s=0.0)
    steps = []
    prev = 0.0
    for k in range(int(2.0 / dt)):
        if k % int(PUBLISH_HZ / TARGET_HZ) == 0:
            s.set_target(18, 0.8 * np.sin(2 * np.pi * 0.5 * k * dt))  # 0.5 Hz arm wave
        v = s.tick()[18]
        steps.append(abs(v - prev))
        prev = v
    # Largest single-tick change stays well under a raw 30 Hz jump (~0.08 rad).
    assert max(steps) < 0.02


def test_engage_ramp_starts_slow():
    dt = 1.0 / PUBLISH_HZ
    s = CommandShaper(np.zeros(29), dt, max_speed=5.0, engage_s=1.5)
    s.set_target(18, 2.0)
    first = s.tick()[18]
    assert first <= 0.21 * 5.0 * dt   # 20 % of the cap on the first tick
