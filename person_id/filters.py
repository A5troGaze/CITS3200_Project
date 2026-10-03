"""
Landmark smoothing and per-limb visibility gating.

OneEuroFilter: Casiez et al. 2012 "1 Euro Filter". A first-order low-pass
whose cutoff rises with speed: heavy smoothing when the person holds still
(kills MediaPipe jitter), little lag when they move fast. Each output is a
convex blend of the new sample and the previous output, so it can never
overshoot the input.

SegmentGate: MediaPipe always returns all 33 landmarks, guessing the ones
it cannot see. A segment whose landmarks are below the visibility
threshold is never passed through. The last trusted value is held for
hold_s seconds, then eased to neutral (upright torso, arm hanging) over
ease_s seconds.
"""

import math

import numpy as np

from mediapipe_to_gmr import SEGMENT_LANDMARKS, Segments, neutral_segments


def _alpha(dt, cutoff):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-euro filter over an (N, D) array of points; the speed that drives
    the adaptive cutoff is each point's Euclidean speed (units/s)."""

    def __init__(self, min_cutoff=0.5, beta=4.0, d_cutoff=0.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.x = None
        self.dx = None
        self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        if dt <= 0:
            return self.x.copy()
        dx = (x - self.x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True) if x.ndim > 1 else np.abs(self.dx)
        cutoff = self.min_cutoff + self.beta * speed
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


def _nlerp(a, b, s):
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
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([ra, rb]))
    return Slerp([0.0, 1.0], rots)([s]).as_matrix()[0]


def _smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class SegmentGate:
    """Hold-then-ease visibility gate over Segments."""

    def __init__(self, hold_s=0.5, ease_s=1.0):
        self.hold_s = hold_s
        self.ease_s = ease_s
        self.neutral = neutral_segments()
        self.reset()

    def reset(self):
        self.last = {}      # segment -> last trusted value
        self.seen_t = {}    # segment -> time it was last trusted

    @staticmethod
    def _get(seg, key):
        if key == "torso":
            return seg.torso
        side, part = key.split("_")
        return (seg.upper if part == "upper" else seg.fore)[side]

    @staticmethod
    def _set(seg, key, value):
        if key == "torso":
            seg.torso = value
            return
        side, part = key.split("_")
        (seg.upper if part == "upper" else seg.fore)[side] = value

    def update(self, measured, t):
        """Returns (gated Segments, {segment: "live" | "hold" | "ease" | "neutral"})."""
        out = measured.copy()
        status = {}
        for key in SEGMENT_LANDMARKS:
            if measured.valid.get(key, False):
                value = self._get(measured, key)
                self.last[key] = value.copy()
                self.seen_t[key] = t
                status[key] = "live"
                continue
            neutral = self._get(self.neutral, key)
            if key not in self.last:
                self._set(out, key, neutral.copy())
                status[key] = "neutral"
                continue
            age = t - self.seen_t[key]
            if age <= self.hold_s:
                self._set(out, key, self.last[key].copy())
                status[key] = "hold"
                continue
            s = _smoothstep((age - self.hold_s) / self.ease_s) if self.ease_s > 0 else 1.0
            if key == "torso":
                value = _slerp_matrix(self.last[key], neutral, s)
            else:
                value = _nlerp(self.last[key], neutral, s)
            self._set(out, key, value)
            status[key] = "ease" if s < 1.0 else "neutral"
        return out, status
