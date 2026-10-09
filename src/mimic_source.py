'''
mimic_source.py

The mimic source: takes one camera frame, finds the person, and returns the
arm angles for the robot to copy.

    BGR frame -> MediaPipe PoseLandmarker -> leader tracking -> MimicPipeline
              -> RobotCommand(joint_targets={motor_index: radians})

It does no I/O of its own (no camera, window or DDS). run.py owns those.

This file used to be called: person_id_brain.py
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import math

from command_source import CommandSource, RobotCommand
from pose_tracking import (
    LeaderTracker,
    bbox_centroid,
    build_landmarker,
    landmarks_to_bbox,
    resolve_model_path,
    torso_histogram,
)


#----------------------------------------------------------
# Mimic source
#----------------------------------------------------------
class MimicSource(CommandSource):   # Renamed from 'PersonIdBrain'
    '''Command source for mimic mode: tracks one person in the camera frame
    and turns their arm movements into joint targets for the robot.

    step(frame_bgr, t) expects t in seconds and returns a RobotCommand.
    joint_targets holds {G1 motor index: angle in rad}; base_velocity is
    always None because this source never commands walking.

    With num_people=1 the only person detected is selected automatically.
    For several people, call select_at(x, y) to pick the leader.

    input_flipped=True is for run.py's shared frame, which has already been
    mirrored with cv2.flip(frame, 1). The source un-flips it before pose
    detection so left and right keep their correct meaning.

    detector, landmarker and pipeline can be passed in (used for testing).
    '''

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

        # leader tracker (kept as kwargs so reset() can rebuild it)
        self._tracker_kwargs = {
            "leader_lost_frames": int(leader_lost_frames),
            "max_match_frac": float(max_match_frac),
        }
        self.tracker = LeaderTracker(**self._tracker_kwargs)
        self.engaged = False                # becomes True once a leader has been locked
        self._pending_click = None          # leader selection waiting for the next frame
        self._last_timestamp_ms = None

        # MediaPipe pose model: use the one passed in, otherwise build our own
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
            self._owns_landmarker = True    # so close() knows it must release it

        # retargeting pipeline (imported here so it only loads when needed)
        if pipeline is None:
            from mimic_pipeline import MimicPipeline

            pipeline = MimicPipeline(
                mirror=mirror,
                waist=waist,
                legs=legs,
                min_visibility=min_visibility,
            )
        self.pipeline = pipeline

        # results of the last frame, for diagnostics
        self.last_detections = []
        self.last_leader = None
        self.last_landmarks_world_m = None
        self.last_mimic = None
        self.last_output = RobotCommand()
        self.last_timestamp_ms = None

    @property
    def locked(self):
        '''True once a leader has been selected.'''
        return self.tracker.locked

    def select_at(self, x, y):
        '''Request multi-person leader selection at pixel coordinates x, y.'''
        self._pending_click = (float(x), float(y))

    def reset(self):
        '''Reset leader/filter state when mimic mode becomes active.

        The MediaPipe landmarker stays alive and its timestamp is not rewound:
        run.py's clock keeps increasing across mode switches, and MediaPipe
        requires timestamps that only go up.'''
        self.tracker = LeaderTracker(**self._tracker_kwargs)
        self.pipeline.reset()
        self.engaged = False
        self._pending_click = None
        self.last_detections = []
        self.last_leader = None
        self.last_landmarks_world_m = None
        self.last_mimic = None
        self.last_output = RobotCommand()

    def close(self):
        '''Release the MediaPipe landmarker if this source created it.'''
        if self._owns_landmarker and self.landmarker is not None:
            close = getattr(self.landmarker, "close", None)
            if close is not None:
                close()
        self.landmarker = None

    #------------------------------------------------------
    # Helpers
    #------------------------------------------------------
    def _timestamp_ms(self, t):
        '''Seconds -> whole milliseconds, forced to always increase (MediaPipe requires it).'''
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
        '''Run pose detection on one frame (or the injected test detector).'''
        if self._detector is not None:
            return self._detector(frame_bgr, timestamp_ms)

        # imported here so they only load when real detection is used
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)    # OpenCV is BGR, MediaPipe wants RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(image, timestamp_ms)

    @staticmethod
    def _landmark_value(lm, key, default=None):
        '''Read a field from a landmark whether it is a dict or an object.'''
        if isinstance(lm, dict):
            return lm.get(key, default)
        return getattr(lm, key, default)

    @classmethod
    def _world_as_dicts(cls, world):
        '''World landmarks -> list of {x, y, z, visibility} dicts for MimicPipeline.'''
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

    #------------------------------------------------------
    # One frame
    #------------------------------------------------------
    def step(self, frame_bgr, t) -> RobotCommand:
        '''Process one caller-owned BGR frame and return a RobotCommand.'''
        if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
            raise ValueError("frame_bgr must be an image-like array")

        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame_bgr has invalid dimensions")

        # run.py flips the frame once for its display and gesture recogniser.
        # Undo that here, so pose tracking sees the original camera image and
        # left/right are not silently swapped.
        processing_frame = frame_bgr[:, ::-1].copy() if self.input_flipped else frame_bgr

        # detect people in the frame
        frame_diag = math.hypot(frame_w, frame_h)
        timestamp_ms = self._timestamp_ms(t)
        result = self._detect(processing_frame, timestamp_ms)
        pose_landmarks = getattr(result, "pose_landmarks", None) or []
        worlds = getattr(result, "pose_world_landmarks", None) or []

        # one entry per detected person: landmarks, bounding box, centre, torso colour histogram
        detections = []
        for i, landmarks in enumerate(pose_landmarks):
            bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)
            detections.append(
                {
                    "landmarks": landmarks,
                    "world_landmarks": worlds[i] if i < len(worlds) else None,
                    "bbox": bbox,
                    "centroid": bbox_centroid(bbox),
                    # the histogram is only needed to tell several people apart
                    "hist": torso_histogram(processing_frame, landmarks, frame_w, frame_h)
                    if self.num_people > 1
                    else None,
                }
            )

        # choose the leader: automatically if only one person, otherwise by click
        if self.num_people == 1:
            if not self.tracker.locked and detections:
                self.tracker.handle_click(*detections[0]["centroid"], detections)
        elif self._pending_click is not None:
            self.tracker.handle_click(*self._pending_click, detections)
            self._pending_click = None

        # follow the leader across frames and get their world landmarks
        leader = self.tracker.update(detections, frame_diag=frame_diag)
        world = leader["world_landmarks"] if leader is not None else None
        landmarks_world_m = self._world_as_dicts(world)

        # Once a leader has been acquired, keep stepping during temporary loss
        # so MimicPipeline can hold and then ease the arms to neutral.
        self.engaged = self.engaged or self.tracker.locked
        mimic = self.pipeline.step(landmarks_world_m, float(t)) if self.engaged else None

        # remember this frame's results and return the arm targets
        self.last_detections = detections
        self.last_leader = leader
        self.last_landmarks_world_m = landmarks_world_m
        self.last_mimic = mimic
        self.last_output = RobotCommand(
            joint_targets={} if mimic is None else dict(mimic.targets),
            base_velocity=None,
        )
        return self.last_output