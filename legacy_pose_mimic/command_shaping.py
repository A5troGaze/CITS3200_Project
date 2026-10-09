"""
Turns ~30 Hz vision targets into a smooth high-rate joint command.

Per control tick (default 500 Hz):
  1. first-order low-pass toward the latest target (time constant tau_s),
     which turns the 30 Hz staircase into a smooth curve without the
     overshoot a second-order filter could add;
  2. per-joint velocity cap (max_speed rad/s, never above the joint's own
     limit in g1_joint_limits.py);
  3. during the first engage_s seconds the cap ramps up from 20 %, so the
     robot never snaps from its start pose to wherever the person is.

The previous controller used only a 1 rad/s cap, which made a raised arm
lag the person by more than a second. With tau 0.05 s and 5 rad/s the command
reaches 95 % of a new target in ~0.15 s for normal arm motion (tested in
tests/test_command_shaping.py).

No DDS here, so it is unit-testable and shared by the sim publisher,
the arm_sdk publisher and the offline demo renderer.
"""

import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS

_VEL_LIMITS = np.zeros(29)
for _lim in G1_29DOF_JOINT_LIMITS.values():
    _VEL_LIMITS[_lim.index] = _lim.velocity


def step_toward(current, target, max_step):
    """Move each element of current toward target by at most max_step."""
    current = np.asarray(current, dtype=float)
    delta = np.clip(np.asarray(target, dtype=float) - current, -max_step, max_step)
    return current + delta


class CommandShaper:
    def __init__(self, start_q, dt, tau_s=0.05, max_speed=5.0, engage_s=1.5):
        self.q = np.asarray(start_q, dtype=float).copy()
        self.target = self.q.copy()
        self.dt = float(dt)
        self.alpha = 1.0 - np.exp(-self.dt / tau_s) if tau_s > 0 else 1.0
        self.max_speed = np.minimum(float(max_speed), _VEL_LIMITS)
        self.engage_s = engage_s
        self.t = 0.0

    def set_target(self, index, value):
        self.target[index] = value

    def tick(self):
        """Advance one control period; returns the new command (29,)."""
        self.t += self.dt
        ramp = 1.0 if self.engage_s <= 0 else min(1.0, 0.2 + 0.8 * self.t / self.engage_s)
        desired = self.q + self.alpha * (self.target - self.q)
        self.q = step_toward(self.q, desired, self.max_speed * ramp * self.dt)
        return self.q.copy()
