'''
command_source.py

Defines the common interface for the "command sources" that run.py can drive
(currently the mimic source). A command source takes one camera frame and
returns the command it wants the robot to carry out. It does no I/O of its
own (no camera, window or DDS), run.py and the controller handle that.

This file used to be called: brain.py
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional, Tuple


#----------------------------------------------------------
# Robot command
#----------------------------------------------------------
@dataclass
class RobotCommand:     # Renamed from 'BrainOutput'
    # motor index -> target angle in rad (arms/waist). May be empty.
    joint_targets: dict = field(default_factory=dict)
    # (vx, vy, vyaw) for locomotion, or None if this source doesn't walk.
    base_velocity: Optional[Tuple[float, float, float]] = None


#----------------------------------------------------------
# Command source interface
#----------------------------------------------------------
class CommandSource:    # Renamed from 'Brain'
    def step(self, frame_bgr, t) -> RobotCommand:
        '''Process one camera frame (BGR, as read by cv2) at time t (seconds)
        and return the command for the robot.'''
        raise NotImplementedError   # every command source must override this

    def reset(self) -> None:
        '''Called when this source becomes the active one.'''
        # default: nothing to reset


#----------------------------------------------------------
# Placeholder source
#----------------------------------------------------------
class PlaceholderMimicSource(CommandSource):    # Renamed from 'PlaceholderMimicBrain'
    '''Fallback used by run.py when MimicSource can't be loaded.
    Does nothing: no joint targets, no walking.'''

    def step(self, frame_bgr, t):
        return RobotCommand()    # empty command, so the robot just keeps balancing