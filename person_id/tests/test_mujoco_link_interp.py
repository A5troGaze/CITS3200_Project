import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS
from mujoco_link import step_toward

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
        delta = np.abs(new_published - published)
        assert np.all(delta <= max_step + 1e-9), f"tick {tick}: delta {delta.max()} exceeds max_step"
        published = new_published
