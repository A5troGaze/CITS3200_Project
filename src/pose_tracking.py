'''
pose_tracking.py

Shared helpers for the pose (mimic) pipeline: finding the pose model, building
the MediaPipe pose landmarker, bounding boxes, leader tracking (which person
to copy), and camera/video helpers.

cv2 and mediapipe are imported inside the functions that need them, so this
module can be imported without a camera, a window or those libraries installed.

This file used to be called: pose_common.py
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import math
import os
import time


#----------------------------------------------------------
# Model location
#----------------------------------------------------------
# Models live in the Dependencies folder, which sits next to the project folder
# (outside it). Found relative to this file, so the project can be installed anywhere.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "..", "Dependencies"))
POSE_MODEL_PATH = os.path.join(DEPENDENCIES_DIR, "Models", "pose_landmarker.task")  # MediaPipe pose model (not in the repo)


#----------------------------------------------------------
# Pose skeleton
#----------------------------------------------------------
# MediaPipe's skeletal connections for the 33 pose landmarks (pairs of landmark
# indices joined by a line). Hardcoded because Tasks-API-only MediaPipe installs
# don't always expose the legacy `solutions` submodule.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


#----------------------------------------------------------
# Model setup
#----------------------------------------------------------
def resolve_model_path(cli_value=None):
    '''Path to the pose model: cli_value if given, otherwise the default location.'''
    if cli_value:
        return cli_value
    return POSE_MODEL_PATH


def build_landmarker(model_path, num_people, min_detection_confidence,
                      min_presence_confidence, min_tracking_confidence):
    '''MediaPipe PoseLandmarker in VIDEO mode. num_people=1 gives the old
    single-person behaviour; >1 gives multi-person detection.'''
    # imported here so MediaPipe only loads when a landmarker is actually built
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

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


#----------------------------------------------------------
# Bounding boxes
#----------------------------------------------------------
def landmarks_to_bbox(landmarks, frame_w, frame_h, padding=20):
    '''Turn normalized landmarks into a pixel-space bounding box, clamped to frame edges.'''
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    x_min = max(int(min(xs)) - padding, 0)
    y_min = max(int(min(ys)) - padding, 0)
    x_max = min(int(max(xs)) + padding, frame_w)
    y_max = min(int(max(ys)) + padding, frame_h)
    return x_min, y_min, x_max, y_max


def bbox_centroid(bbox):
    '''Centre point (x, y) of a bounding box.'''
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def bbox_area(bbox):
    '''Area of a bounding box in pixels.'''
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


#----------------------------------------------------------
# Drawing and timestamps
#----------------------------------------------------------
def draw_skeleton(frame, landmarks, frame_w, frame_h, color=(0, 200, 0)):
    '''Draw connected skeleton lines plus joint dots.'''
    import cv2

    # landmarks are fractions of the frame (0-1), convert to pixel positions
    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)
    for (x, y) in points:
        cv2.circle(frame, (x, y), 3, color, -1)


def make_timestamp(is_live_camera, start_time, frame_index, fps):
    '''Timestamp in ms since start. A live camera uses the wall clock (frames
    don't arrive at a fixed rate); a video file uses fps so replay timing
    stays consistent.'''
    if is_live_camera:
        return int((time.time() - start_time) * 1000)
    return int((frame_index / fps) * 1000)


#----------------------------------------------------------
# Camera and video
#----------------------------------------------------------
def open_capture(input_arg, width, height):
    '''Open a camera index or video file. Returns
    (cap, frame_w, frame_h, fps, is_live_camera).

    MJPG is only requested for a live camera: webcam passthrough is often
    bandwidth-limited, and forcing MJPG on a video file can make some
    backends refuse to open it at all.
    '''
    import cv2

    # a number means a camera index, anything else is a video file path
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0     # some cameras report 0, assume 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # cap.set() can fail silently if the driver rejects the request, so check
    # what was actually negotiated instead of assuming it worked.
    if is_live_camera and (frame_w, frame_h) != (width, height):
        print(f"Note: requested {width}x{height}, camera delivers {frame_w}x{frame_h} "
              "(`v4l2-ctl --device=/dev/video0 --list-formats-ext` lists its modes).")

    return cap, frame_w, frame_h, fps, is_live_camera


def open_video_writer(output_path, fps, frame_w, frame_h):
    '''Open an annotated-output VideoWriter, or None if output_path is falsy.
    Checked with isOpened() so a bad codec or path fails loudly instead of
    silently producing an empty file.'''
    import cv2

    if not output_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video '{output_path}'.")
    return writer


#----------------------------------------------------------
# FPS counter
#----------------------------------------------------------
class FpsCounter:
    '''Rolling FPS counter, refreshed once a second.'''

    def __init__(self):
        self.last_check = time.time()
        self.frame_count = 0
        self.display_fps = 0.0

    def tick(self):
        '''Call once per frame; returns the latest FPS value.'''
        self.frame_count += 1
        now = time.time()
        if now - self.last_check >= 1.0:
            self.display_fps = self.frame_count / (now - self.last_check)
            self.frame_count = 0
            self.last_check = now
        return self.display_fps


#----------------------------------------------------------
# Torso appearance (to tell people apart)
#----------------------------------------------------------
TORSO_LANDMARKS = (11, 12, 24, 23)  # shoulder L, shoulder R, hip R, hip L (polygon order)
HIST_BINS = (16, 8)                 # hue x saturation


def torso_histogram(frame_bgr, landmarks, frame_w, frame_h):
    '''Normalised 2-D HSV (hue x saturation) histogram of the torso region
    (shoulder/hip quadrilateral, shrunk 15 % toward its centre so arms and
    background at the edges are mostly excluded). Returns a flat float32
    array summing to 1, or None if the torso is too small or off-frame.
    Clothing colour is a cheap, robust cue to tell two people apart when
    their boxes cross.'''
    import cv2
    import numpy as np

    # torso corners in pixels, shrunk towards the centre and kept inside the frame
    pts = np.array([[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in TORSO_LANDMARKS],
                   dtype=np.float32)
    centre = pts.mean(axis=0)
    pts = centre + 0.85 * (pts - centre)
    pts[:, 0] = np.clip(pts[:, 0], 0, frame_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, frame_h - 1)
    if cv2.contourArea(pts) < 150:
        return None     # torso too small to be reliable

    # colour histogram of just the pixels inside the torso
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, list(HIST_BINS), [0, 180, 0, 256]).flatten()
    total = hist.sum()
    if total <= 0:
        return None
    return (hist / total).astype(np.float32)    # normalise so it sums to 1


def histogram_distance(a, b):
    '''Bhattacharyya/Hellinger distance between two normalised histograms:
    0 = identical, 1 = no overlap.'''
    if a is None or b is None:
        return None
    bc = float(sum(math.sqrt(max(x, 0.0) * max(y, 0.0)) for x, y in zip(a, b)))
    return math.sqrt(max(0.0, 1.0 - bc))


#----------------------------------------------------------
# Leader tracking
#----------------------------------------------------------
class LeaderTracker:
    '''Click-to-select leader tracking.

    Each frame, every detection is scored against the leader by
      cost = (1 - w) * (distance to predicted centroid / match radius)
             + w * appearance distance (torso HSV histogram, 0..1)
    and the cheapest detection within the match radius wins. The predicted
    centroid adds the leader's recent velocity, and the match radius is a
    fraction of the frame diagonal (max_match_frac, so it scales with
    resolution) unless an absolute max_match_distance in pixels is given.
    A detection whose clothing clearly differs (appearance distance above
    max_appearance_distance) is never taken as the leader, even if it is
    the closest. That is what stops identity swapping when two people cross.

    `detections` are dicts with at least "bbox" and "centroid", plus
    optional "hist" (torso_histogram). Extra keys pass through untouched.
    Detections without a histogram fall back to distance-only matching.
    '''

    def __init__(self, max_match_distance=None, leader_lost_frames=15, max_match_frac=0.2,
                 appearance_weight=0.5, max_appearance_distance=0.6, hist_update=0.1,
                 use_velocity=True):
        self.max_match_distance = max_match_distance
        self.max_match_frac = max_match_frac
        self.leader_lost_frames = leader_lost_frames
        self.appearance_weight = appearance_weight
        self.max_appearance_distance = max_appearance_distance
        self.hist_update = hist_update
        self.use_velocity = use_velocity

        # tracking state
        self.locked = False             # True once a leader is selected
        self.centroid = None            # leader's last known position
        self.velocity = (0.0, 0.0)      # leader's recent movement per frame
        self.hist = None                # leader's clothing histogram
        self.lost_frames = 0            # frames since the leader was last matched

    def _radius(self, frame_diag):
        '''How far (pixels) a detection may be from the predicted position and still match.'''
        if self.max_match_distance is not None:
            return float(self.max_match_distance)
        if frame_diag is None:
            return 120.0
        return self.max_match_frac * frame_diag

    def handle_click(self, x, y, detections):
        '''Select whichever detection contains (x, y), preferring the
        smallest (most specific) box when several overlap. Returns the
        selected detection, or None if the click missed everyone (in which
        case the current leader, if any, is left untouched).'''
        best_area, best = None, None
        for d in detections:
            x_min, y_min, x_max, y_max = d["bbox"]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                area = bbox_area(d["bbox"])
                if best_area is None or area < best_area:
                    best_area, best = area, d
        if best is not None:
            # lock on and start tracking from this detection
            self.locked = True
            self.centroid = best["centroid"]
            self.velocity = (0.0, 0.0)
            self.hist = best.get("hist")
            self.lost_frames = 0
        return best

    def predicted_centroid(self):
        '''Where the leader should be now (last position plus velocity).'''
        if not self.use_velocity:
            return self.centroid
        k = self.lost_frames + 1    # the longer the leader is lost, the further it may have moved
        return (self.centroid[0] + k * self.velocity[0], self.centroid[1] + k * self.velocity[1])

    def update(self, detections, frame_diag=None):
        '''Match the locked leader against this frame's detections.

        Must be called every frame, even with no detections, so that
        lost_frames advances and the lock is released after
        leader_lost_frames unmatched frames.
        '''
        if not self.locked:
            return None

        radius = self._radius(frame_diag)
        px, py = self.predicted_centroid()
        w = self.appearance_weight

        # score every detection and keep the cheapest one that qualifies
        matched, best_cost = None, None
        for d in detections or []:
            dist = math.hypot(d["centroid"][0] - px, d["centroid"][1] - py)
            if dist > radius:
                continue                # too far from where the leader should be
            app = histogram_distance(self.hist, d.get("hist"))
            if app is None:
                cost = dist / radius    # no clothing info: distance only
            else:
                if app > self.max_appearance_distance:
                    continue            # clearly different clothing: not the leader
                cost = (1 - w) * dist / radius + w * app
            if best_cost is None or cost < best_cost:
                best_cost, matched = cost, d

        if matched is not None:
            # leader found: update position, velocity and clothing histogram
            cx, cy = matched["centroid"]
            steps = self.lost_frames + 1
            vx, vy = (cx - self.centroid[0]) / steps, (cy - self.centroid[1]) / steps
            self.velocity = (0.5 * self.velocity[0] + 0.5 * vx, 0.5 * self.velocity[1] + 0.5 * vy)
            self.centroid = (cx, cy)
            h = matched.get("hist")
            if h is not None:
                if self.hist is None:
                    self.hist = h
                else:
                    a = self.hist_update    # slowly blend in the new histogram
                    self.hist = [(1 - a) * x + a * y for x, y in zip(self.hist, h)]
            self.lost_frames = 0
        else:
            # leader not found this frame; give up after leader_lost_frames in a row
            self.lost_frames += 1
            if self.lost_frames > self.leader_lost_frames:
                self.locked = False
                self.centroid = None
                self.hist = None
                self.velocity = (0.0, 0.0)
                self.lost_frames = 0
        return matched