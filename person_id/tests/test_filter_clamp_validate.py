import pytest

from g1_joint_limits import G1_29DOF_JOINT_LIMITS, clamp_position, validate_command
from retarget_upper_body import TargetFilter

JOINT = "left_elbow_joint"
LIM = G1_29DOF_JOINT_LIMITS[JOINT]


def test_rate_limiter_never_exceeds_velocity_times_dt():
    f = TargetFilter(alpha=1.0)  # alpha=1 disables smoothing to isolate the rate limiter
    dt = 0.001  # small enough that velocity*dt stays within the joint's own range

    f.step({JOINT: 0.0}, dt)  # establish a starting point
    result = f.step({JOINT: LIM.upper}, dt)  # try to jump straight to the limit

    max_step = LIM.velocity * dt
    assert result[JOINT] == pytest.approx(max_step, abs=1e-6)


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
