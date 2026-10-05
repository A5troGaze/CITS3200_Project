"""Reusable per-frame Person ID / mimic brain.

This module owns the *per-frame* Person ID work only:

    BGR frame -> MediaPipe PoseLandmarker -> leader tracking -> MimicPipeline
              -> {G1 motor index: target angle in radians}

It deliberately does not open a camera, create an OpenCV window, publish DDS,
or own a main loop.  That lets gesture_control.py (or any other owner) keep one
camera/main loop and call ``PersonIdBrain.step(frame, t)`` while in mimic mode.

Default integration behaviour is ``num_people=1``: the only detected person is
automatically selected, so mimic mode does not depend on the old click-to-select
preview.  Multi-person mode is still supported through ``select_at(x, y)``;
the owner process decides how/where those coordinates are collected.
"""

import math

from pose_common import (
    LeaderTracker,
    bbox_centroid,
    build_landmarker,
    landmarks_to_bbox,
    resolve_model_path,
    torso_histogram,
)


class PersonIdBrain:
    """Camera/window-free wrapper around the existing Person ID pipeline.

    Parameters mirror the relevant options from ``leader_pose.py``.  In normal
    integrated use the defaults are enough: one person, arms only, anatomical
    mapping.

    ``step(frame_bgr, t)`` expects ``t`` in seconds and returns a dictionary of
    motor-index -> angle-radians targets.  Before a leader has been acquired it
    returns an empty dictionary.  Once engaged, temporary leader loss is passed
    to MimicPipeline as ``None`` so its existing hold/ease-to-neutral behaviour
    is preserved.

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
        *,
        detector=None,
        landmarker=None,
        pipeline=None,
    ):
        if num_people < 1:
            raise ValueError("num_people must be at least 1")

        self.num_people = int(num_people)
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
            # Lazy import keeps importing PersonIdBrain cheap/testable on a
            # machine that does not have GMR/Pinocchio installed.
            from mimic_pipeline import MimicPipeline

            pipeline = MimicPipeline(
                mirror=mirror,
                waist=waist,
                legs=legs,
                min_visibility=min_visibility,
            )
        self.pipeline = pipeline

        # Introspection for the legacy leader_pose UI and for integration
        # diagnostics.  step() itself still returns only the arm targets.
        self.last_detections = []
        self.last_leader = None
        self.last_landmarks_world_m = None
        self.last_mimic = None
        self.last_timestamp_ms = None

    @property
    def locked(self):
        return self.tracker.locked

    def select_at(self, x, y):
        """Request multi-person leader selection at pixel coordinates ``x,y``.

        The request is applied to detections from the next ``step``.  It is a
        no-op for the usual ``num_people=1`` integration, where the only person
        is auto-selected.
        """
        self._pending_click = (float(x), float(y))

    def reset(self):
        """Reset leader lock and mimic/filter state for a mode switch.

        MediaPipe's VIDEO landmarker itself is intentionally kept alive and its
        timestamp is *not* rewound; detect_for_video requires monotonically
        increasing timestamps across the lifetime of the landmarker.
        """
        self.tracker = LeaderTracker(**self._tracker_kwargs)
        self.pipeline.reset()
        self.engaged = False
        self._pending_click = None
        self.last_detections = []
        self.last_leader = None
        self.last_landmarks_world_m = None
        self.last_mimic = None

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
        # Gesture/main loops may deliver the same wall-clock millisecond twice.
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
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
        out = []
        for lm in world:
            out.append(
                {
                    "x": float(cls._landmark_value(lm, "x")),
                    "y": float(cls._landmark_value(lm, "y")),
                    "z": float(cls._landmark_value(lm, "z")),
                    "visibility": float(cls._landmark_value(lm, "visibility", 1.0)),
                }
            )
        return out

    def step(self, frame_bgr, t):
        """Process one BGR frame and return current G1 arm targets.

        ``frame_bgr`` is supplied/owned by the caller.  This method never opens
        a capture device, calls imshow/waitKey, or publishes to DDS.
        """
        if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
            raise ValueError("frame_bgr must be an image-like array")

        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame_bgr has invalid dimensions")
        frame_diag = math.hypot(frame_w, frame_h)
        timestamp_ms = self._timestamp_ms(t)

        result = self._detect(frame_bgr, timestamp_ms)
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
                    "hist": torso_histogram(frame_bgr, landmarks, frame_w, frame_h)
                    if self.num_people > 1
                    else None,
                }
            )

        # Integration default: no click/UI is needed.  With one-person pose
        # detection, acquire whichever person is present as soon as they appear.
        if self.num_people == 1:
            if not self.tracker.locked and detections:
                self.tracker.handle_click(*detections[0]["centroid"], detections)
        elif self._pending_click is not None:
            self.tracker.handle_click(*self._pending_click, detections)
            self._pending_click = None

        leader = self.tracker.update(detections, frame_diag=frame_diag)
        world = leader["world_landmarks"] if leader is not None else None
        landmarks_world_m = self._world_as_dicts(world)

        # Preserve leader_pose.py's behaviour: once a leader has ever been
        # acquired, keep stepping the pipeline during temporary loss so its
        # visibility gate can hold and then ease the arms to neutral.
        self.engaged = self.engaged or self.tracker.locked
        mimic = self.pipeline.step(landmarks_world_m, float(t)) if self.engaged else None

        self.last_detections = detections
        self.last_leader = leader
        self.last_landmarks_world_m = landmarks_world_m
        self.last_mimic = mimic

        return {} if mimic is None else dict(mimic.targets)
