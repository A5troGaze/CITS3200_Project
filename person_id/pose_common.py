"""
Shared building blocks for the person_id camera pipeline.

Everything here is a pure function or a plain-data class: nothing opens a
camera or a window at import time, so this module can be unit-tested without
a display, a webcam, or MediaPipe's native camera bindings.

Used by leader_pose.py (and previously duplicated ~80% verbatim across
pose_test.py, pose_test-world_landmarks.py and pose_test-multi-world-landmarks.py,
which this module replaces).
"""

import math
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Standardised model location (see SETUP.md). Override with --model or the
# CITS3200_MODELS_DIR environment variable if your model lives elsewhere.
DEFAULT_MODELS_DIR = os.path.expanduser("~/CITS3200/Dependencies/Models")
DEFAULT_MODEL_FILENAME = "pose_landmarker.task"

# MediaPipe's documented skeletal connections for the 33 pose landmarks —
# drawing these (not just dots) is what makes the overlay actually read as
# a skeleton in the demo, matching what the client asked for.
# Hardcoded here (rather than pulled from mp.solutions.pose) because
# Tasks-API-only MediaPipe installs don't always expose the legacy
# `solutions` submodule. This is a fixed fact about the model's output
# structure, not something that should ever be a config option.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def resolve_model_path(cli_value=None):
    """Standardised model path resolution (defect 7): --model overrides
    everything; otherwise CITS3200_MODELS_DIR env var + the standard
    filename; otherwise the default ~/CITS3200/Dependencies/Models location.
    """
    if cli_value:
        return cli_value
    models_dir = os.environ.get("CITS3200_MODELS_DIR", DEFAULT_MODELS_DIR)
    return os.path.join(models_dir, DEFAULT_MODEL_FILENAME)


def build_landmarker(model_path, num_people, min_detection_confidence,
                      min_presence_confidence, min_tracking_confidence):
    """MediaPipe PoseLandmarker in VIDEO mode. num_people=1 gives the old
    single-person behaviour; >1 gives multi-person detection."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_people,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def landmarks_to_bbox(landmarks, frame_w, frame_h, padding=20):
    """Turn normalized landmarks into a pixel-space bounding box, clamped to frame edges."""
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    x_min = max(int(min(xs)) - padding, 0)
    y_min = max(int(min(ys)) - padding, 0)
    x_max = min(int(max(xs)) + padding, frame_w)
    y_max = min(int(max(ys)) + padding, frame_h)
    return x_min, y_min, x_max, y_max


def bbox_centroid(bbox):
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def draw_skeleton(frame, landmarks, frame_w, frame_h, color=(0, 200, 0)):
    """Draw connected skeleton lines plus joint dots."""
    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 3, color, -1)


def make_timestamp(is_live_camera, start_time, frame_index, fps):
    """Standardised timestamp: ms since start (defect 9). Live camera derives
    it from the wall clock since frame arrival isn't tied to a fixed rate;
    a video file derives it from fps so replay timing stays consistent."""
    if is_live_camera:
        return int((time.time() - start_time) * 1000)
    return int((frame_index / fps) * 1000)


def open_capture(input_arg, width, height):
    """Open a camera index or video file. Returns
    (cap, frame_w, frame_h, fps, is_live_camera).

    MJPG is only requested for a live camera (defect 2): VirtualBox webcam
    passthrough is often bandwidth-limited, and forcing MJPG on a video file
    input can make some backends refuse to open it at all.
    """
    video_source = int(input_arg) if str(input_arg).isdigit() else input_arg
    is_live_camera = isinstance(video_source, int)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open input '{input_arg}'. If this is a webcam, check "
            "VirtualBox Devices -> Webcams is enabled for this VM session, and "
            "run `ls -l /dev/video*` to confirm the device node exists."
        )

    if is_live_camera:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # cap.set() can fail silently if the driver rejects the request — check
    # what was actually negotiated instead of assuming it worked.
    if (frame_w, frame_h) != (width, height):
        print(f"WARNING: requested {width}x{height} but the input is "
              f"actually delivering {frame_w}x{frame_h}. The driver ignored "
              "the request — run `v4l2-ctl --device=/dev/video0 "
              "--list-formats-ext` to see what it actually supports.")

    return cap, frame_w, frame_h, fps, is_live_camera


def open_video_writer(output_path, fps, frame_w, frame_h):
    """Open an annotated-output VideoWriter, or None if output_path is falsy.
    Checked with isOpened() (defect 4) so a bad codec/path fails loudly
    instead of silently producing an empty file."""
    if not output_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video '{output_path}'.")
    return writer


class FpsCounter:
    """Rolling FPS counter, refreshed once a second."""

    def __init__(self):
        self.last_check = time.time()
        self.frame_count = 0
        self.display_fps = 0.0

    def tick(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_check >= 1.0:
            self.display_fps = self.frame_count / (now - self.last_check)
            self.frame_count = 0
            self.last_check = now
        return self.display_fps


class LeaderTracker:
    """Click-to-select leader tracking by nearest bounding-box centroid.

    `detections` passed to handle_click/update is a list of dicts, each with
    at least a "bbox" (x_min, y_min, x_max, y_max) and "centroid" (x, y) key
    (build these with landmarks_to_bbox/bbox_centroid over each frame's
    detections; extra keys such as "landmarks" are carried through untouched
    since these methods return the whole matched dict).
    """

    def __init__(self, max_match_distance=120.0, leader_lost_frames=15):
        self.max_match_distance = max_match_distance
        self.leader_lost_frames = leader_lost_frames
        self.locked = False
        self.centroid = None
        self.lost_frames = 0

    def handle_click(self, x, y, detections):
        """Select whichever detection contains (x, y), preferring the
        smallest (most specific) box when several overlap. Returns the
        selected detection, or None if the click missed everyone (in which
        case the current leader, if any, is left untouched)."""
        best_area, best = None, None
        for d in detections:
            x_min, y_min, x_max, y_max = d["bbox"]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                area = bbox_area(d["bbox"])
                if best_area is None or area < best_area:
                    best_area, best = area, d
        if best is not None:
            self.locked = True
            self.centroid = best["centroid"]
            self.lost_frames = 0
        return best

    def update(self, detections):
        """Match the locked leader against this frame's detections.

        Must be called every frame regardless of whether detections is
        empty (defect 1): pose_test.py's `if locked and detections:` guard
        skipped this on empty frames, so lost_frames never advanced and the
        leader never timed out. Counting every unmatched frame unconditionally
        is the correct behaviour used here.
        """
        if not self.locked:
            return None

        matched = None
        if detections:
            best_dist, best = None, None
            for d in detections:
                dx = d["centroid"][0] - self.centroid[0]
                dy = d["centroid"][1] - self.centroid[1]
                dist = math.hypot(dx, dy)
                if best_dist is None or dist < best_dist:
                    best_dist, best = dist, d
            if best_dist is not None and best_dist <= self.max_match_distance:
                matched = best

        if matched is not None:
            self.centroid = matched["centroid"]
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            if self.lost_frames > self.leader_lost_frames:
                self.locked = False
                self.centroid = None
                self.lost_frames = 0
        return matched
