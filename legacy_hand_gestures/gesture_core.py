# Convert Landmarks to vectors and classify them

import numpy as np  # Import NumPy
import cv2


#== Define hand model ==========================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17) 
]



#== Draw 21-point hand skeleton ================================================
def draw_skeleton(frame, result):
    if not result.hand_landmarks:
        return
    h, w, _ = frame.shape                                                       # Get height and width (discard channel count) from frame
    for hand_landmarks in result.hand_landmarks:                                # For each hand (Only 1 currently):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]      # Get real pixel position
        for start_idx, end_idx in HAND_CONNECTIONS:                             # For each connected pair in defined hand model
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2) # Draw a line from the first to the second point
        for point in points:                                                    # Likewise:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)                        # draw a circle at each point



#== Take list of 21 landmark objects MediaPipe gives============================
def landmarks_to_vector(hand_landmarks):
    '''
    Convert hand landmarks into a normalized flat vector. Same gestures results in a similar vector, regardless of distance/position from camera.
    '''
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])            # Build NumPy array from raw landmarks. Extract the x, y, z values 

    wrist = coords[0]                                                           # Subtract x, y, z triple of wrist (index 0) from all coords
    coords = coords - wrist                                                     # All 'coords' are now expressed as 'relative to the wrist' rather than relative to the camer frame.

    #== Standardise coordinates ======================
    scale = np.linalg.norm(coords[9])                                           # Calculate the vector length from the wrist to the middle knuckle ([9])
    if scale > 0:                                                               # Defense against 0 division
        coords = coords / scale                                                 # Divide every coord by the reference distance of the wrist to the knuckle

    return coords.flatten()                                                     # Collapse 2D array into single 1D Array of 63 numbers


'''
== Takes: =====================================================================
- vector (63 number fingerprint from landmarks_to_vector)
- registry (dictionary of gesture names and stored sample vectors)
- threshold (distance cutoff, default of 1.5)
'''
def classify(vector, registry, threshold= 1.5):
    '''
    Compare live hand vectors against stored samples.
    Return (gesture_name, distance) or (None, distance) if nothing is close enough to a stored gesture.
    '''

    best_name, best_dist = None, float("inf")                                   # Initialise tracking variables
    for name, samples in registry.items():                                      # Go through every stored gesture in the registry
        for sample in samples:
            sample = np.array(sample)                                           # Convert current sample back into a NumPy array
            dist = np.linalg.norm(vector - sample)                              # Subtract stored sample from live vector and return a single number
            if dist < best_dist:                                                # If sample distance is smaller than best so far:
                best_dist = dist                                                # Sample becomes new closest match
                best_name = name                                                # Update gesture name

    if best_dist < threshold:                                                   # SAFETY CHECK: Only trust and return if the closes distance is actually under the maximum threshold. If it is too far away return None instead.
        return best_name, best_dist
    return None, best_dist

