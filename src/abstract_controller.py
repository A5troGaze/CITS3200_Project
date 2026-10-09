'''
abstract_controller.py

Defines the interface (the "contract") that every robot backend must follow.
run.py only ever talks to a controller through these methods, so it never
needs to know whether it is driving the simulator or the real robot.

Backends that implement this interface:
    SimController   (simulation_controller.py) - MuJoCo simulator
    RealController  (real_controller.py)       - physical G1 robot
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from abc import ABC, abstractmethod


#----------------------------------------------------------
# Interface
#----------------------------------------------------------
class AbstractController(ABC):

    #------------------------------------------------------
    # Required methods (every backend MUST implement these)
    #------------------------------------------------------
    @abstractmethod
    def init(self):
        '''One-time setup: open connections, create publishers, etc.
        Called once before start().'''

    @abstractmethod
    def start(self):
        '''Bring the robot to a state where it can accept gestures
        (e.g. stand up / get ready to walk). Called once after init().'''

    @abstractmethod
    def set_gesture(self, gesture_name):
        '''Gesture mode: called every camera frame with the name of the
        currently recognised gesture (e.g. "turn_left", "move_forward_left_hand"),
        or None if no gesture is recognised. Also called with None when
        switching to mimic mode, so the robot stops walking.
        The backend turns this into a walking command.'''

    @abstractmethod
    def stop(self):
        '''Shut down safely: stop walking and release any connections.
        Called once when run.py exits.'''

    #------------------------------------------------------
    # Optional methods (backends MAY override these)
    #------------------------------------------------------
    def on_mode_change(self, mode):
        '''Called by run.py when the user switches between "gesture" and
        "mimic" mode (key "m").
        Default: do nothing.'''

    def set_joint_targets(self, targets):
        '''Mimic mode: {G1 motor index: angle in rad} from the mimic source,
        called once per camera frame.
        Default: ignore them (backends that cannot move the arms,
        e.g. RealController).'''