"""Reusable per-frame Person ID / mimic brain.

This module owns the per-frame Person ID work only:

    BGR frame -> MediaPipe PoseLandmarker -> leader tracking -> MimicPipeline
              -> BrainOutput(joint_targets={motor_index: radians})

It deliberately does not open a camera, create an OpenCV window, publish DDS,
or own a main loop.  gesture_control.py can therefore remain the single owner
of the camera, preview window, mode switch, and robot I/O.

The gesture Integration branch flips the shared camera frame before handing it
to the active brain.  Set ``input_flipped=True`` when PersonIdBrain is called
from that loop; the brain will undo that display flip before pose detection so
Person ID keeps the same left/right convention as standalone leader_pose.py.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Use the gesture team's shared Brain/BrainOutput when it is available.  The
# fallbacks keep the Person ID branch runnable/testable before the Integration
# branch is merged.
try:  # package import after the Integration branch is merged
    from gesture_recognition.brain import Brain, BrainOutput
except ImportError:
    try:  # gesture_control.py is normally executed from gesture_recognition/
        from brain import Brain, BrainOutput
    except ImportError:
        @dataclass
        class BrainOutput:
            joint_targets: dict = field(default_factory=dict)
            base_velocity: Optional[Tuple[float, float, float]] = None

        class Brain:
            def step(self, frame_bgr, t) -> BrainOutput:
                raise NotImplementedError

            def reset(self) -> None:
                pass

try:
    from .pose_common import (
        LeaderTracker,
        bbox_centroid,
        build_landmarker,
        landmarks_to_bbox,
        resolve_model_path,
        torso_histogram,
    )
except ImportError:  # standalone scripts/tests run with person_id/ on sys.path
    from pose_common import (
        LeaderTracker,
        bbox_centroid,
        build_landmarker,
        landmarks_to_bbox,
        resolve_model_path,
        torso_histogram,
    )


class PersonIdBrain(Brain):
    """Camera/window/DDS-free wrapper around the existing Person ID pipeline.

    ``step(frame_bgr, t)`` expects ``t`` in seconds and returns a BrainOutput.
    ``joint_targets`` contains G1 motor-index -> angle-radians targets and
    ``base_velocity`` is always ``None`` because Person ID does not command
    locomotion.

    With ``num_people=1`` the only detected person is selected automatically,
    so integrated mimic mode needs no click UI.  Multi-person callers can use
    ``select_at(x, y)``.

    ``input_flipped=True`` is for gesture_control.py's shared frame, which has
    already passed through ``cv2.flip(frame, 1)``.  Standalone leader_pose.py
    passes raw camera frames and therefore leaves it False.

    ``detector`` and ``pipeline`` are injectable for unit tests.  A detector is
    a callable ``detector(frame_bgr, timestamp_ms)`` returning a MediaPipe-like
    result with ``pose_landmarks`` and ``pose_world_landmarks``.
    """

    def __init__(
        self,
        model_path=None,
        num_people=1,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        max_match_frac=0.2,
        leader_lost_frames=15,
        mirror=False,
        waist="off",
        legs=False,
        min_visibility=0.5,
        input_flipped=False,
        *,
        detector=None,
        landmarker=None,
        pipeline=None,
    ):
        if num_people < 1:
            raise ValueError("num_people must be at least 1")

        self.num_people = int(num_people)
        self.input_flipped = bool(input_flipped)
        self._tracker_kwargs = {
            "leader_lost_frames": int(leader_lost_frames),
            "max_match_frac": float(max_match_frac),
        }
        self.tracker = LeaderTracker(**self._tracker_kwargs)
        self.engaged = False
        self._pending_click = None
        self._last_timestamp_ms = None

        self._owns_landmarker = False
        self.landmarker = landmarker
        self._detector = detector
        if self._detector is None and self.landmarker is None:
            self.landmarker = build_landmarker(
                resolve_model_path(model_path),
                self.num_people,
                min_detection_confidence,
                min_presence_confidence,
                min_tracking_confidence,
            )
            self._owns_landmarker = True

        if pipeline is None:
            try:
                from .mimic_pipeline import MimicPipeline
            except ImportError:
                from mimic_pipeline import MimicPipeline

            pipeline = MimicPipeline(
                mirror=mirror,
                waist=waist,
                legs=legs,
                min_visibility=min_visibility,
            )
        self.pipeline = pipeline

        # Introspection for the standalone leader_pose UI and diagnostics.
        self.last_detections = []
        self.last_leader = None
        self.last_landmarks_world_m = None
        self.last_mimic = None
        self.last_output = BrainOutput()
        self.last_timestamp_ms = None

    @property
    def locked(self):
        return self.tracker.locked

    def select_at(self, x, y):
        """Request multi-person leader selection at pixel coordinates ``x,y``."""
        self._pending_click = (float(x), float(y))

    def reset(self):
        """Reset leader/filter state when mimic mode becomes active.

        The MediaPipe landmarker remains alive and its VIDEO timestamp is not
        rewound.  gesture_control.py's synthetic clock keeps increasing across
        mode switches, and MediaPipe requires monotonically increasing times.
        """
        self.tracker = LeaderTracker(**self._tracker_kwargs)
        self.pipeline.reset()
        self.engaged = False
        self._pending_click = None
        self.last_detections = []
        self.last_leader = None
        self.last_landmarks_world_m = None
        self.last_mimic = None
        self.last_output = BrainOutput()

    def close(self):
        """Release the MediaPipe landmarker if this brain created it."""
        if self._owns_landmarker and self.landmarker is not None:
            close = getattr(self.landmarker, "close", None)
            if close is not None:
                close()
        self.landmarker = None

    def _timestamp_ms(self, t):
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t must be finite seconds")
        ts = int(round(t * 1000.0))
        if self._last_timestamp_ms is not None and ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        self.last_timestamp_ms = ts
        return ts

    def _detect(self, frame_bgr, timestamp_ms):
        if self._detector is not None:
            return self._detector(frame_bgr, timestamp_ms)

        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(image, timestamp_ms)

    @staticmethod
    def _landmark_value(lm, key, default=None):
        if isinstance(lm, dict):
            return lm.get(key, default)
        return getattr(lm, key, default)

    @classmethod
    def _world_as_dicts(cls, world):
        if world is None:
            return None
        return [
            {
                "x": float(cls._landmark_value(lm, "x")),
                "y": float(cls._landmark_value(lm, "y")),
                "z": float(cls._landmark_value(lm, "z")),
                "visibility": float(cls._landmark_value(lm, "visibility", 1.0)),
            }
            for lm in world
        ]

    def step(self, frame_bgr, t) -> BrainOutput:
        """Process one caller-owned BGR frame and return a BrainOutput."""
        if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
            raise ValueError("frame_bgr must be an image-like array")

        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame_bgr has invalid dimensions")

        # gesture_control.py flips once for its display/gesture recogniser.  Do
        # pose tracking in the original camera orientation so Person ID's
        # established anatomical/mirror semantics do not silently swap sides.
        processing_frame = frame_bgr[:, ::-1].copy() if self.input_flipped else frame_bgr

        frame_diag = math.hypot(frame_w, frame_h)
        timestamp_ms = self._timestamp_ms(t)
        result = self._detect(processing_frame, timestamp_ms)
        pose_landmarks = getattr(result, "pose_landmarks", None) or []
        worlds = getattr(result, "pose_world_landmarks", None) or []

        detections = []
        for i, landmarks in enumerate(pose_landmarks):
            bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)
            detections.append(
                {
                    "landmarks": landmarks,
                    "world_landmarks": worlds[i] if i < len(worlds) else None,
                    "bbox": bbox,
                    "centroid": bbox_centroid(bbox),
                    "hist": torso_histogram(processing_frame, landmarks, frame_w, frame_h)
                    if self.num_people > 1
                    else None,
                }
            )

        if self.num_people == 1:
            if not self.tracker.locked and detections:
                self.tracker.handle_click(*detections[0]["centroid"], detections)
        elif self._pending_click is not None:
            self.tracker.handle_click(*self._pending_click, detections)
            self._pending_click = None

        leader = self.tracker.update(detections, frame_diag=frame_diag)
        world = leader["world_landmarks"] if leader is not None else None
        landmarks_world_m = self._world_as_dicts(world)

        # Once a leader has been acquired, keep stepping during temporary loss
        # so MimicPipeline can hold and then ease the arms to neutral.
        self.engaged = self.engaged or self.tracker.locked
        mimic = self.pipeline.step(landmarks_world_m, float(t)) if self.engaged else None

        self.last_detections = detections
        self.last_leader = leader
        self.last_landmarks_world_m = landmarks_world_m
        self.last_mimic = mimic
        self.last_output = BrainOutput(
            joint_targets={} if mimic is None else dict(mimic.targets),
            base_velocity=None,
        )
        return self.last_output
