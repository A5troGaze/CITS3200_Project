'''
joint_smoothing.py

Smooths the arm targets coming from the camera (about 30 per second) into a
steady stream of commands for the robot. Each tick it:
    1. eases the command towards the latest target (low-pass filter)
    2. caps how fast each joint can move (rad/s)
    3. starts slowly after engaging, so the arms never snap to the first target

Used by ArmSdkPublisher. No DDS in here, so it can be tested on its own.

This file used to be called: command_shaping.py
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS


#----------------------------------------------------------
# Joint velocity limits
#----------------------------------------------------------
# max speed (rad/s) of each of the 29 motors, taken from g1_joint_limits.py
_VEL_LIMITS = np.zeros(29)
for _lim in G1_29DOF_JOINT_LIMITS.values():
    _VEL_LIMITS[_lim.index] = _lim.velocity


#----------------------------------------------------------
# Helpers
#----------------------------------------------------------
def step_toward(current, target, max_step):
    '''Move each element of current toward target by at most max_step.'''
    current = np.asarray(current, dtype=float)
    delta = np.clip(np.asarray(target, dtype=float) - current, -max_step, max_step)
    return current + delta


#----------------------------------------------------------
# Command shaper
#----------------------------------------------------------
class JointSmoother:    # Renamed from 'CommandShaper'
    def __init__(self, start_q, dt, tau_s=0.05, max_speed=5.0, engage_s=1.5):
        self.q = np.asarray(start_q, dtype=float).copy()    # current command (starts at the real pose)
        self.target = self.q.copy()                         # latest target from the camera
        self.dt = float(dt)                                 # seconds per tick
        # smoothing factor per tick, derived from the time constant tau_s
        # (no smoothing if tau_s is 0)
        self.alpha = 1.0 - np.exp(-self.dt / tau_s) if tau_s > 0 else 1.0
        # speed cap per joint: max_speed, but never above the joint's own limit
        self.max_speed = np.minimum(float(max_speed), _VEL_LIMITS)
        self.engage_s = engage_s                            # seconds of slow start after engaging
        self.t = 0.0                                        # time since start

    def set_target(self, index, value):
        '''Set the target angle (rad) for one motor.'''
        self.target[index] = value

    def tick(self):
        '''Advance one control period; returns the new command (29,).'''
        self.t += self.dt
        # speed cap multiplier: starts at 20% and rises to 100% over engage_s
        ramp = 1.0 if self.engage_s <= 0 else min(1.0, 0.2 + 0.8 * self.t / self.engage_s)
        # 1) ease towards the target
        desired = self.q + self.alpha * (self.target - self.q)
        # 2) limit how far the command may move this tick
        self.q = step_toward(self.q, desired, self.max_speed * ramp * self.dt)
        return self.q.copy()