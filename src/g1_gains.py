'''
g1_gains.py

PD gains (stiffness kp, damping kd) for the G1's motors. Copied from
Unitree's own examples in unitree_sdk2_python. Kept in a separate file so
they can be used without importing unitree_sdk2py.
'''

#----------------------------------------------------------
# Per-joint gains (G1 motor-index order)
#----------------------------------------------------------
KP = [
    60, 60, 60, 100, 40, 40,      # left leg
    60, 60, 60, 100, 40, 40,      # right leg
    60, 40, 40,                   # waist
    40, 40, 40, 40, 40, 40, 40,   # left arm
    40, 40, 40, 40, 40, 40, 40,   # right arm
]
KD = [
    1, 1, 1, 2, 1, 1,             # left leg
    1, 1, 1, 2, 1, 1,             # right leg
    1, 1, 1,                      # waist
    1, 1, 1, 1, 1, 1, 1,          # left arm
    1, 1, 1, 1, 1, 1, 1,          # right arm
]

#----------------------------------------------------------
# arm_sdk gains (same value for every arm joint)
#----------------------------------------------------------
ARM_SDK_KP = 60.0
ARM_SDK_KD = 1.5