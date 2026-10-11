'''
pose_filtering.py

Cleans up the MediaPipe landmarks before they reach the robot:
    - OneEuroFilter: smooths jitter when you hold still, with little lag
      when you move fast
    - SegmentGate: if a limb isn't visible, holds its last good value
      briefly, then eases it back to a neutral pose

This file used to be called: filters.py
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import math

import numpy as np

from RUN.pose_to_target import SEGMENT_LANDMARKS, neutral_segments


#----------------------------------------------------------
# One Euro filter (smoothing)
#----------------------------------------------------------
def _alpha(dt, cutoff):
    '''Smoothing factor (0-1) for a given time step and cutoff frequency.'''
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    '''One Euro filter (Casiez et al. 2012) over an (N, D) array of points.
    The cutoff rises with speed: heavy smoothing when still, light when moving.
    The output is a blend of the new sample and the previous output, so it
    never overshoots the input.'''

    def __init__(self, min_cutoff=0.5, beta=4.0, d_cutoff=0.5):
        self.min_cutoff = float(min_cutoff)     # smoothing when still (lower = smoother)
        self.beta = float(beta)                 # how fast the cutoff rises with speed
        self.d_cutoff = float(d_cutoff)         # smoothing applied to the speed estimate
        self.reset()

    def reset(self):
        '''Forget all history (next call starts fresh).'''
        self.x = None       # previous filtered value
        self.dx = None      # previous smoothed velocity
        self.t = None       # time of previous sample

    def __call__(self, x, t):
        '''Filter sample x taken at time t (seconds); returns the smoothed copy.'''
        x = np.asarray(x, dtype=float)

        # first sample: nothing to smooth against, pass it straight through
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()

        # time went backwards or didn't move: keep the previous output
        dt = t - self.t
        if dt <= 0:
            return self.x.copy()

        # estimate velocity, then smooth it
        dx = (x - self.x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx

        # each point's speed (units/s) sets its cutoff: faster = less smoothing
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True) if x.ndim > 1 else np.abs(self.dx)
        cutoff = self.min_cutoff + self.beta * speed
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)

        # blend the new sample with the previous output
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


#----------------------------------------------------------
# Interpolation helpers (used when easing back to neutral)
#----------------------------------------------------------
def _nlerp(a, b, s):
    '''Blend unit vector a towards b by fraction s, result stays unit length.'''
    v = (1 - s) * a + s * b
    n = np.linalg.norm(v)
    if n < 1e-6:
        # Opposite directions: go via any perpendicular.
        perp = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1.0, 0])
        v = perp
        n = np.linalg.norm(v)
    return v / n


def _slerp_matrix(ra, rb, s):
    '''Smoothly rotate rotation matrix ra towards rb by fraction s.'''
    # SciPy imported here, only when a torso needs easing
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([ra, rb]))
    return Slerp([0.0, 1.0], rots)([s]).as_matrix()[0]


def _smoothstep(s):
    '''Maps 0-1 to an S-curve (slow start, slow end) so the ease isn't abrupt.'''
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


#----------------------------------------------------------
# Visibility gate
#----------------------------------------------------------
class SegmentGate:
    '''MediaPipe always returns every landmark, guessing the ones it can't see.
    This gate never passes a segment through when it isn't visible:
        live    -> segment is visible, passed through
        hold    -> not visible, last good value held for hold_s seconds
        ease    -> after that, eased towards neutral over ease_s seconds
        neutral -> fully back at the neutral pose (upright torso, arms hanging)'''

    def __init__(self, hold_s=0.5, ease_s=1.0):
        self.hold_s = hold_s
        self.ease_s = ease_s
        self.neutral = neutral_segments()   # the pose to fall back to
        self.reset()

    def reset(self):
        '''Forget all history.'''
        self.last = {}      # segment -> last trusted value
        self.seen_t = {}    # segment -> time it was last trusted

    @staticmethod
    def _get(seg, key):
        '''Read one segment out of a Segments object by its key.'''
        if key == "torso":
            return seg.torso
        side, part = key.split("_")
        return (seg.upper if part == "upper" else seg.fore)[side]

    @staticmethod
    def _set(seg, key, value):
        '''Write one segment into a Segments object by its key.'''
        if key == "torso":
            seg.torso = value
            return
        side, part = key.split("_")
        (seg.upper if part == "upper" else seg.fore)[side] = value

    def update(self, measured, t):
        '''Returns (gated Segments, {segment: "live" | "hold" | "ease" | "neutral"}).'''
        out = measured.copy()
        status = {}
        for key in SEGMENT_LANDMARKS:
            # visible: trust it and remember it
            if measured.valid.get(key, False):
                value = self._get(measured, key)
                self.last[key] = value.copy()
                self.seen_t[key] = t
                status[key] = "live"
                continue

            # never seen yet: nothing to hold, use neutral
            neutral = self._get(self.neutral, key)
            if key not in self.last:
                self._set(out, key, neutral.copy())
                status[key] = "neutral"
                continue

            # recently lost: hold the last good value
            age = t - self.seen_t[key]
            if age <= self.hold_s:
                self._set(out, key, self.last[key].copy())
                status[key] = "hold"
                continue

            # lost for a while: ease from the last good value to neutral
            s = _smoothstep((age - self.hold_s) / self.ease_s) if self.ease_s > 0 else 1.0
            if key == "torso":
                value = _slerp_matrix(self.last[key], neutral, s)   # torso is a rotation matrix
            else:
                value = _nlerp(self.last[key], neutral, s)          # limbs are direction vectors
            self._set(out, key, value)
            status[key] = "ease" if s < 1.0 else "neutral"
        return out, status