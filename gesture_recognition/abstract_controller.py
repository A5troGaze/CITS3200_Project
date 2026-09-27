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