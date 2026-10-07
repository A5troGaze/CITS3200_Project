"""gesture_control.py's "onboard" backend: gesture mode walks, mimic mode
stops walking and drives the arms. No DDS: the arm publisher and the
velocity publisher are replaced by recorders."""
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("vgamepad", types.ModuleType("vgamepad"))
sys.path.append(str(Path(__file__).resolve().parents[2] / "gesture_recognition"))

from onboard_sim_controller import OnboardSimController  # noqa: E402


class FakeArms:
    def __init__(self):
        self.calls = []

    def engage(self):
        self.calls.append("engage")

    def disengage(self):
        self.calls.append("disengage")
        return True

    def set_targets(self, targets):
        self.calls.append(("targets", dict(targets)))

    def stop(self):
        self.calls.append("stop")


@pytest.fixture
def ctl():
    c = OnboardSimController()
    c.arms = FakeArms()
    c.sent = []
    c._publish_velocity = lambda vx, vy, wz: c.sent.append((vx, vy, wz))
    return c


def test_gesture_mode_walks_and_ignores_arm_targets(ctl):
    ctl.set_gesture("move_forward_left_hand")
    assert ctl._current_velocity() == (0.5, 0.0, 0.0)
    ctl.set_joint_targets({15: 0.3})
    assert ctl.arms.calls == []


def test_mimic_mode_stops_walking_and_drives_arms(ctl):
    ctl.set_gesture("move_forward_left_hand")
    ctl.on_mode_change("mimic")
    assert ctl.mode == "mimic"
    assert ctl.current_gesture is None
    assert ctl.arms.calls == ["engage"]
    ctl.set_joint_targets({15: 0.3, 22: -0.2})
    ctl.set_joint_targets({})           # nobody in view: keep the last pose
    assert ctl.arms.calls[-1] == ("targets", {15: 0.3, 22: -0.2})


def test_back_to_gesture_returns_the_arms(ctl):
    ctl.on_mode_change("mimic")
    ctl.on_mode_change("gesture")
    assert ctl.mode == "gesture"
    assert ctl.arms.calls == ["engage", "disengage"]
    ctl.set_joint_targets({15: 0.3})
    assert ctl.arms.calls == ["engage", "disengage"]


def test_run_loop_publishes_zero_in_mimic_mode(ctl):
    ctl.set_gesture("move_left")
    ctl.on_mode_change("mimic")
    ctl._stop_event.wait = lambda timeout: setattr(ctl, "_stopped", True)
    ctl._run_loop()
    assert ctl.sent == [(0.0, 0.0, 0.0)]


def test_stop_halts_and_releases_arms(ctl):
    ctl.velocity_pub = object()
    ctl.on_mode_change("mimic")
    ctl.stop()
    assert ctl.sent[-1] == (0.0, 0.0, 0.0)
    assert ctl.arms.calls[-2:] == ["disengage", "stop"]
