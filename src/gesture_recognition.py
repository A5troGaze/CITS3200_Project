'''
gesture_recognition.py

Turns MediaPipe hand landmarks into a normalised vector, matches it against
the stored gestures (classify), and draws the hand skeleton on the camera frame.

This file used to be called: gesture_core.py
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
import numpy as np
import cv2


#----------------------------------------------------------
# Hand model
#----------------------------------------------------------
# pairs of landmark indices (0-20) that are joined by a line
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17)                                  # palm edge (wrist to pinky base)
]


#----------------------------------------------------------
# Drawing
#----------------------------------------------------------
def draw_skeleton(frame, result):
    '''Draw the 21-point hand skeleton onto the camera frame.'''
    if not result.hand_landmarks:
        return
    h, w, _ = frame.shape                       # frame height and width (channel count not needed)
    for hand_landmarks in result.hand_landmarks:    # each detected hand (only 1 currently)
        # landmarks are fractions of the frame (0-1), convert to pixel positions
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        # green line for every connected pair
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        # red dot on every landmark
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)


#----------------------------------------------------------
# Landmarks -> vector
#----------------------------------------------------------
def landmarks_to_vector(hand_landmarks):
    '''Convert the 21 MediaPipe landmarks into a normalised flat vector (63 numbers).
    The same gesture gives a similar vector regardless of distance or position
    relative to the camera.'''
    # x, y, z of every landmark as a (21, 3) array
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    # position relative to the wrist (landmark 0) instead of the camera frame
    wrist = coords[0]
    coords = coords - wrist

    # scale by the wrist-to-middle-knuckle distance (landmark 9) so hand size doesn't matter
    scale = np.linalg.norm(coords[9])
    if scale > 0:                       # avoid dividing by zero
        coords = coords / scale

    return coords.flatten()             # (21, 3) -> flat array of 63 numbers


#----------------------------------------------------------
# Classification
#----------------------------------------------------------
# vector:    63-number fingerprint from landmarks_to_vector
# registry:  {gesture name: list of stored sample vectors}
# threshold: maximum distance to count as a match (default 1.5)
def classify(vector, registry, threshold=1.5):
    '''Compare the live hand vector against the stored samples.
    Returns (gesture_name, distance), or (None, distance) if nothing is close enough.'''
    best_name, best_dist = None, float("inf")   # closest match so far

    for name, samples in registry.items():      # every stored gesture
        for sample in samples:                  # every sample of that gesture
            sample = np.array(sample)           # stored samples are plain lists
            dist = np.linalg.norm(vector - sample)  # distance between live vector and sample
            if dist < best_dist:                # closest so far: remember it
                best_dist = dist
                best_name = name

    # only trust the closest match if it is within the threshold, otherwise no gesture
    if best_dist < threshold:
        return best_name, best_dist
    return None, best_dist