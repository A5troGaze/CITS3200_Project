# Convert Landmarks to vectors and classify them

import numpy as np  # Import NumPy

# Take list of 21 landmark objects MediaPipe gives
def landmarks_to_vector(hand_landmarks):
    '''
    Convert hand landmarks into a normalized flat vector. Same gestures results in a similar vector, regardless of distance/position from camera.
    '''
    #== Build NumPy array from raw landmarks. Extract the x, y, z values ====
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    #== Subtract x, y, z triple of wrist (index 0) from all coords =====
    wrist = coords[0]
    coords = coords - wrist
    # All 'coords' are now expressed as 'relative to the wrist' rather than relative to the camer frame.

    #== Standardise coordinates ======================
    scale = np.linalg.norm(coords[9])   # Calculate the vector length from the wrist to the middle knuckle ([9])
    if scale > 0:                       # Defense against 0 division
        coords = coords / scale         # Divide every coord by the reference distance of the wrist to the knuckle

    #== Collapse 2D array into single 1D Array of 63 numbers =======
    return coords.flatten()


'''
Takes: 
- vector (63 number fingerprint from landmarks_to_vector)
- registry (dictionary of gesture names and stored sample vectors)
- threshold (distance cutoff, default of 0.3)
'''
def classify(vector, registry, threshold=0.9):
    '''
    Compare live hand vectors against stored samples.
    Return (gesture_name, distance) or (None, distance) if nothing is close enough to a stored gesture.
    '''

    best_name, best_dist = None, float("inf")   # Initialise tracking variables
    for name, samples in registry.items():      # Go through every stored gesture in the registry
        for sample in samples:
            sample = np.array(sample)           # Convert current sample back into a NumPy array
            dist = np.linalg.norm(vector - sample)  # Subtract stored sample from live vector and return a single number
            if dist < best_dist:                # If sample distance is smaller than best so far:
                best_dist = dist                # Sample becomes new closest match
                best_name = name                # Update gesture name

    if best_dist < threshold:                   # SAFETY CHECK: Only trust and return if the closes distance is actually under the maximum threshold. If it is too far away return None instead.
        return best_name, best_dist
    return None, best_dist
