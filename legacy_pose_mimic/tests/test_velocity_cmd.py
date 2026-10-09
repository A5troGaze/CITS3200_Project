import math

from velocity_cmd import LIMITS, VelocityCommand


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_zero_until_first_message():
    assert VelocityCommand().get() == (0.0, 0.0, 0.0)


def test_latest_command_is_used():
    clock = FakeClock()
    v = VelocityCommand(clock=clock)
    v.set(0.5, -0.3, 0.7)
    clock.now = 0.2
    assert v.get() == (0.5, -0.3, 0.7)


def test_stops_when_sender_goes_quiet():
    clock = FakeClock()
    v = VelocityCommand(timeout_s=0.5, clock=clock)
    v.set(0.5, 0.0, 0.0)
    clock.now = 0.6
    assert v.get() == (0.0, 0.0, 0.0)


def test_clipped_to_limits_and_non_finite_is_zero():
    v = VelocityCommand()
    v.set(5.0, -5.0, math.nan)
    assert v.get() == (LIMITS[0][1], LIMITS[1][0], 0.0)


def test_gesture_table_is_inside_limits():
    # Every command the gesture team sends must reach the policy unchanged.
    import sys
    import types
    from pathlib import Path

    sys.modules.setdefault("vgamepad", types.ModuleType("vgamepad"))
    sys.path.append(str(Path(__file__).resolve().parents[2] / "gesture_recognition"))
    import simulation_controller as sc

    commands = list(sc.GESTURE_CMD.values()) + list(sc.TURN_FORWARD_CMD.values()) \
        + list(sc.TURN_BACKWARD_CMD.values())
    for cmd in commands:
        for value, (lo, hi) in zip(cmd, LIMITS):
            assert lo <= value <= hi, cmd
