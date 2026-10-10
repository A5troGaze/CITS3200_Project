import numpy as np
import pytest

from command_shaping import CommandShaper, step_toward
from g1_joint_limits import G1_29DOF_JOINT_LIMITS, clamp_position, validate_command

JOINT = "left_elbow_joint"
LIM = G1_29DOF_JOINT_LIMITS[JOINT]


def test_rate_limiter_never_exceeds_speed_times_dt():
    dt = 0.002
    shaper = CommandShaper(np.zeros(29), dt, tau_s=0.0, max_speed=5.0, engage_s=0.0)  # no smoothing
    shaper.set_target(LIM.index, LIM.upper)  # try to jump straight to the limit
    q = shaper.tick()
    assert q[LIM.index] == pytest.approx(5.0 * dt, abs=1e-9)


def test_step_toward_is_symmetric_and_stops_at_target():
    assert step_toward([0.0], [1.0], 0.3)[0] == pytest.approx(0.3)
    assert step_toward([0.0], [-1.0], 0.3)[0] == pytest.approx(-0.3)
    assert step_toward([0.0], [0.1], 0.3)[0] == pytest.approx(0.1)


def test_clamp_never_exceeds_table_bounds():
    assert clamp_position(JOINT, LIM.upper + 999) == LIM.upper
    assert clamp_position(JOINT, LIM.lower - 999) == LIM.lower


def test_validate_command_catches_out_of_range_value():
    violations = validate_command({JOINT: LIM.upper + 1.0})
    assert violations
    assert any(JOINT in v for v in violations)


def test_validate_command_passes_in_range_value():
    violations = validate_command({JOINT: (LIM.lower + LIM.upper) / 2})
    assert violations == []
