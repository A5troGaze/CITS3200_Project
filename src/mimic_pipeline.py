'''
mimic_pipeline.py

The full chain that turns one camera frame of body landmarks into G1 arm angles:

    33 MediaPipe landmarks
      -> smoothing (OneEuroFilter)
      -> body-relative bone directions (measure_segments)
      -> visibility gate (hold the last pose, then ease to neutral)
      -> G1 targets (TargetBuilder)
      -> inverse kinematics (GMR)
      -> {G1 motor index: angle in rad}, clamped to the real joint limits

No camera or DDS in here. The caller (the mimic source) feeds it landmarks.
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import time
from dataclasses import dataclass, field

import numpy as np

from RUN.pose_filtering import OneEuroFilter, SegmentGate
from g1_joint_limits import G1_29DOF_JOINT_LIMITS
from gmr_ik_solver import GmrIKSolver, commanded_joint_names
from RUN.pose_to_target import (
    SEGMENT_LANDMARKS,
    TargetBuilder,
    landmarks_to_array,
    measure_segments,
    mirror_landmarks,
    segment_validity,
)


#----------------------------------------------------------
# Result of one frame
#----------------------------------------------------------
@dataclass
class MimicResult:
    targets: dict                    # {motor_index: angle_rad}, commanded joints only
    q: dict                          # {joint_name: angle_rad}, all 29 (clamped)
    status: dict                     # segment -> live/hold/ease/neutral
    timings_ms: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


#----------------------------------------------------------
# Pipeline
#----------------------------------------------------------
class MimicPipeline:
    def __init__(self, mirror=False, waist="off", legs=False, min_visibility=0.5,
                 visibility_hysteresis=0.1, hold_s=0.5, ease_s=1.0,
                 min_cutoff=0.5, beta=4.0, d_cutoff=0.5, smooth=True, ik_solver=None):
        self.mirror = mirror                        # swap left/right so the robot copies like a mirror
        self.min_visibility = min_visibility        # landmarks below this count as not visible
        self.hysteresis = visibility_hysteresis     # dead band around that threshold (stops flicker)
        self.smooth = smooth                        # False skips the One Euro filter

        # the stages of the pipeline
        self.gmr = ik_solver or GmrIKSolver()
        self.builder = TargetBuilder(self.gmr.model, waist=waist)
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.gate = SegmentGate(hold_s=hold_s, ease_s=ease_s)

        # which joints we send targets for, and their G1 motor indices
        self.commanded = commanded_joint_names(waist=waist, legs=legs)
        self.commanded_idx = {name: G1_29DOF_JOINT_LIMITS[name].index for name in self.commanded}

        self._valid = {k: False for k in SEGMENT_LANDMARKS}     # visibility verdict from the previous frame
        self._last_q = None

    def reset(self):
        '''Forget all history (e.g. when the user switches back to mimic mode).'''
        self.filter.reset()
        self.gate.reset()
        self.builder.reset()
        self.gmr.reset()
        self._valid = {k: False for k in SEGMENT_LANDMARKS}

    def _validity(self, arr):
        '''Per-segment visibility with hysteresis, so a landmark hovering at
        the threshold does not flicker the segment between live and hold.'''
        # becoming visible needs the upper threshold, staying visible only the lower one
        hi = segment_validity(arr, self.min_visibility + self.hysteresis / 2)
        lo = segment_validity(arr, self.min_visibility - self.hysteresis / 2)
        self._valid = {k: hi[k] or (self._valid[k] and lo[k]) for k in hi}
        return dict(self._valid)

    def step(self, landmarks, t):
        '''One frame. `landmarks`: 33 world landmarks (dicts, MediaPipe
        objects or a (33, 3|4) array); `t`: seconds (monotonic).'''
        t0 = time.perf_counter()    # wall-clock timer for this frame
        c0 = time.thread_time()     # CPU time for this frame

        if landmarks is None:
            # No leader this frame: every segment is unseen, so the gate
            # holds the last pose, then eases to neutral.
            return self._finish(self._unseen(), t, t0, c0)

        arr = landmarks_to_array(landmarks)

        # a non-finite landmark is treated as invisible, and must not poison the filter
        bad = ~np.all(np.isfinite(arr[:, :3]), axis=1)
        if np.any(bad):
            arr[bad, 3] = 0.0   # visibility 0
            fill = self.filter.x[bad] if self.filter.x is not None else 0.0     # use the previous filtered value
            arr[bad, :3] = fill

        # 1) smooth the landmark positions
        if self.smooth:
            arr[:, :3] = self.filter(arr[:, :3], t)

        # 2) optional left/right mirror
        if self.mirror:
            arr = mirror_landmarks(arr)

        # 3) which segments are visible this frame (with hysteresis)
        valid = self._validity(arr)

        # 4) landmarks -> bone directions; if the geometry is unusable, treat everything as unseen
        try:
            measured = measure_segments(arr, min_visibility=self.min_visibility)
            measured.valid = {k: measured.valid[k] and valid[k] for k in valid}
        except ValueError:
            measured = self._unseen()

        return self._finish(measured, t, t0, c0)

    def _unseen(self):
        '''A Segments object with nothing visible (neutral pose, all invalid).'''
        seg = self.gate.neutral.copy()
        seg.valid = {k: False for k in SEGMENT_LANDMARKS}
        return seg

    def _finish(self, measured, t, t0, c0):
        '''Gate -> targets -> IK -> motor targets, and time each stage.'''
        # visibility gate: live / hold / ease / neutral
        segments, status = self.gate.update(measured, t)

        # segments -> GMR targets
        t1 = time.perf_counter()
        human, weights, info = self.builder.build(segments)

        # GMR inverse kinematics -> joint angles by name
        t2 = time.perf_counter()
        q = self.gmr.solve(human, weights)
        t3 = time.perf_counter()

        # joint name -> G1 motor index, commanded joints only
        targets = {idx: q[name] for name, idx in self.commanded_idx.items()}
        if not all(np.isfinite(v) for v in targets.values()):
            raise ValueError("non-finite joint target")

        self._last_q = q
        info["segments"] = segments
        info["gmr_calls"] = self.gmr.last_calls
        return MimicResult(
            targets=targets, q=q, status=status, info=info,
            timings_ms={
                "adapter": (t1 - t0) * 1000 + (t2 - t1) * 1000,     # everything before GMR
                "gmr": (t3 - t2) * 1000,                            # the IK solve
                "total": (t3 - t0) * 1000,
                "gmr_cpu": self.gmr.last_solve_cpu_ms,
                "total_cpu": (time.thread_time() - c0) * 1000,
            },
        )