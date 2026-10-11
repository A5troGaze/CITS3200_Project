"""
Walking command for the on-board balance policy (sim_standing.py --rl-lab).

The policy inside the simulator (sim_onboard.py) was only ever given a zero
velocity command, so the robot could balance and move its arms but not walk.
This topic lets gesture_control.py's "onboard" backend drive the same policy
in gesture mode, so walking (gestures) and arm mimicry (rt/arm_sdk) run on one
simulator with one balance controller, and switching modes never stops it.

    topic   rt/cmd_vel   (sim only; the real G1 walks through Unitree's own
                          locomotion, see real_controller.py)
    message VelocityCmd_ (vx m/s forward, vy m/s left, vyaw rad/s counter-clockwise)

The values are what the policy sees as its velocity command, i.e. the same
(vx, vy, wz) the gesture team's GESTURE_CMD table already uses.
"""

import time

import numpy as np

TOPIC = "rt/cmd_vel"

# Clipped to what the velocity policy handles comfortably; the gesture table
# stays well inside these.
LIMITS = ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0))

# If the sender stops (crashed, closed), stop walking rather than keep going
# on the last command.
TIMEOUT_S = 0.5


class VelocityCommand:
    """Latest received command, clipped, and zero once it is older than
    TIMEOUT_S. Written from the DDS thread, read from the physics thread
    (a tuple assignment, so no lock is needed)."""

    def __init__(self, timeout_s=TIMEOUT_S, clock=time.monotonic):
        self.timeout_s = timeout_s
        self.clock = clock
        self._latest = None              # ((vx, vy, vyaw), received_at)

    def set(self, vx, vy, vyaw):
        values = []
        for v, (lo, hi) in zip((vx, vy, vyaw), LIMITS):
            v = float(v)
            values.append(float(np.clip(v, lo, hi)) if np.isfinite(v) else 0.0)
        self._latest = (tuple(values), self.clock())

    def on_message(self, msg):
        """rt/cmd_vel VelocityCmd_ callback (DDS thread)."""
        self.set(msg.vx, msg.vy, msg.vyaw)

    def get(self):
        latest = self._latest
        if latest is None or self.clock() - latest[1] > self.timeout_s:
            return (0.0, 0.0, 0.0)
        return latest[0]


def message_type():
    """The DDS message class (imported lazily so this module, and its tests,
    don't need cyclonedds)."""
    global _VelocityCmd
    if _VelocityCmd is None:
        from dataclasses import dataclass

        from cyclonedds.idl import IdlStruct
        from cyclonedds.idl.types import float32

        @dataclass
        class VelocityCmd_(IdlStruct, typename="cits3200.VelocityCmd_"):
            vx: float32
            vy: float32
            vyaw: float32

        _VelocityCmd = VelocityCmd_
    return _VelocityCmd


_VelocityCmd = None
