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