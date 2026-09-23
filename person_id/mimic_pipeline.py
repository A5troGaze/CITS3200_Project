"""
The one person_id retargeting pipeline:

    33 MediaPipe world landmarks
      -> one-euro filter (positions)
      -> measure_segments (body-relative bone directions, --mirror)
      -> visibility gate with hysteresis (hold, then ease to neutral)
      -> TargetBuilder (G1 link-frame targets)
      -> GMR (warm-started IK)
      -> joint name -> G1 motor index, clamped to the real limits.

Used by leader_pose.py (live camera), person_id_replay.py, the demo
recorder and tests/test_mimicry.py, so what is tested is what runs.
No DDS or camera imports here.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from filters import OneEuroFilter, SegmentGate
from g1_joint_limits import G1_29DOF_JOINT_LIMITS
from gmr_retarget import GmrRetargeter, commanded_joint_names
from mediapipe_to_gmr import (
    SEGMENT_LANDMARKS,
    TargetBuilder,
    landmarks_to_array,
    measure_segments,
    mirror_landmarks,
    segment_validity,
)


@dataclass
class MimicResult:
    targets: dict                    # {motor_index: angle_rad}, commanded joints only
    q: dict                          # {joint_name: angle_rad}, all 29 (clamped)
    status: dict                     # segment -> live/hold/ease/neutral
    timings_ms: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class MimicPipeline:
    def __init__(self, mirror=False, waist="3dof", legs=False, min_visibility=0.5,
                 visibility_hysteresis=0.1, hold_s=0.5, ease_s=1.0,
                 min_cutoff=0.5, beta=4.0, d_cutoff=0.5, smooth=True, retargeter=None):
        self.mirror = mirror
        self.min_visibility = min_visibility
        self.hysteresis = visibility_hysteresis
        self.smooth = smooth
        self.gmr = retargeter or GmrRetargeter()
        self.builder = TargetBuilder(self.gmr.model, waist=waist)
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.gate = SegmentGate(hold_s=hold_s, ease_s=ease_s)
        self.commanded = commanded_joint_names(waist=waist, legs=legs)
        self.commanded_idx = {name: G1_29DOF_JOINT_LIMITS[name].index for name in self.commanded}
        self._valid = {k: False for k in SEGMENT_LANDMARKS}
        self._last_q = None

    def reset(self):
        self.filter.reset()
        self.gate.reset()
        self.builder.reset()
        self.gmr.reset()
        self._valid = {k: False for k in SEGMENT_LANDMARKS}

    def _validity(self, arr):
        """Per-segment visibility with hysteresis, so a landmark hovering at
        the threshold does not flicker the segment between live and hold."""
        hi = segment_validity(arr, self.min_visibility + self.hysteresis / 2)
        lo = segment_validity(arr, self.min_visibility - self.hysteresis / 2)
        self._valid = {k: hi[k] or (self._valid[k] and lo[k]) for k in hi}
        return dict(self._valid)

    def step(self, landmarks, t):
        """One frame. `landmarks`: 33 world landmarks (dicts, MediaPipe
        objects or a (33, 3|4) array); `t`: seconds (monotonic)."""
        t0 = time.perf_counter()
        c0 = time.thread_time()
        if landmarks is None:
            # No leader this frame: every segment is unseen, so the gate
            # holds the last pose, then eases to neutral.
            return self._finish(self._unseen(), t, t0, c0)
        arr = landmarks_to_array(landmarks)
        bad = ~np.all(np.isfinite(arr[:, :3]), axis=1)
        if np.any(bad):
            # A non-finite landmark is invisible, and must not poison the filter.
            arr[bad, 3] = 0.0
            fill = self.filter.x[bad] if self.filter.x is not None else 0.0
            arr[bad, :3] = fill
        if self.smooth:
            arr[:, :3] = self.filter(arr[:, :3], t)
        if self.mirror:
            arr = mirror_landmarks(arr)
        valid = self._validity(arr)
        try:
            measured = measure_segments(arr, min_visibility=self.min_visibility)
            measured.valid = {k: measured.valid[k] and valid[k] for k in valid}
        except ValueError:
            measured = self._unseen()
        return self._finish(measured, t, t0, c0)

    def _unseen(self):
        seg = self.gate.neutral.copy()
        seg.valid = {k: False for k in SEGMENT_LANDMARKS}
        return seg

    def _finish(self, measured, t, t0, c0):
        segments, status = self.gate.update(measured, t)
        t1 = time.perf_counter()
        human, weights, info = self.builder.build(segments)
        t2 = time.perf_counter()
        q = self.gmr.solve(human, weights)
        t3 = time.perf_counter()
        targets = {idx: q[name] for name, idx in self.commanded_idx.items()}
        if not all(np.isfinite(v) for v in targets.values()):
            raise ValueError("non-finite joint target")
        self._last_q = q
        info["segments"] = segments
        info["gmr_calls"] = self.gmr.last_calls
        return MimicResult(
            targets=targets, q=q, status=status, info=info,
            timings_ms={
                "adapter": (t1 - t0) * 1000 + (t2 - t1) * 1000,
                "gmr": (t3 - t2) * 1000,
                "total": (t3 - t0) * 1000,
                "gmr_cpu": self.gmr.last_solve_cpu_ms,
                "total_cpu": (time.thread_time() - c0) * 1000,
            },
        )
