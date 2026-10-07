from abc import ABC, abstractmethod

class AbstractGestureController:
    @abstractmethod
    def init(self):
        ''''''

    @abstractmethod
    def start(self):
        ''''''

    @abstractmethod
    def set_gesture(self):
        ''''''
        
    @abstractmethod
    def stop(self):
        ''''''

    def on_mode_change(self, mode):
        '''Called when gesture_control.py switches between "gesture" and
        "mimic" mode. Default: do nothing (e.g. RealController).'''

    def set_joint_targets(self, targets):
        '''Mimic mode: {G1 motor index: angle in rad} from the mimic brain,
        called once per camera frame. Default: ignore them (backends that
        can't move the arms yet, e.g. RealController, SimController).'''