"""
brain.py

Common interface for the "brains" that gesture_control.py's main loop can run.
Only one brain is active at a time (selected by the mode toggle), and the main
loop owns the camera and passes each frame to the active brain.

See person_id/INTEGRATION.md ("Proposed switch: one owner process with a mode
flag") for the design this follows.

A brain does no I/O of its own: no camera, no window, no DDS. It takes a frame
and returns what it wants the robot to do. Whoever owns the output (the
controller / mode manager) decides how to send it.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class BrainOutput:
    # motor index -> target angle in rad (arms/waist). May be empty.
    joint_targets: dict = field(default_factory=dict)
    # (vx, vy, vyaw) for locomotion, or None if this brain doesn't walk.
    base_velocity: Optional[Tuple[float, float, float]] = None


class Brain:
    def step(self, frame_bgr, t) -> BrainOutput:
        """Process one camera frame (BGR, as read by cv2) at time t (seconds)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Called when this brain becomes the active one."""


class PlaceholderMimicBrain(Brain):
    """Stand-in for person_id's brain until PersonIdBrain.step() exists.
    Does nothing: no joint targets, no walking."""

    def step(self, frame_bgr, t):
        return BrainOutput()
