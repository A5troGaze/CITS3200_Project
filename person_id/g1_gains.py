"""
G1 PD gains, motor-index order (see G1_ARM_CONVENTIONS.md section 4).

KP/KD are copied verbatim from unitree_sdk2_python
example/g1/low_level/g1_low_level_example.py (lines 18-32).
ARM_SDK_KP/KD are from example/g1/high_level/g1_arm7_sdk_dds_example.py
(lines 74-75). Kept in their own module so tests and the offline
simulator can use them without importing unitree_sdk2py.
"""

KP = [
    60, 60, 60, 100, 40, 40,      # left leg
    60, 60, 60, 100, 40, 40,      # right leg
    60, 40, 40,                   # waist
    40, 40, 40, 40, 40, 40, 40,   # left arm
    40, 40, 40, 40, 40, 40, 40,   # right arm
]
KD = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
]

ARM_SDK_KP = 60.0
ARM_SDK_KD = 1.5
