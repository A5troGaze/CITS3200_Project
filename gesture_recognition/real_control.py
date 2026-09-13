from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from abstract_controller import AbstractGestureController


class RealController(AbstractGestureController):
    def __init__(self, interface="wlan0"):
        self.interface = interface
        self.current_gesture = None

    def init(self):
        ChannelFactoryInitialize(0, self.interface)
        self.sport_client = LocoClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

    def start(self):
        pass

    def set_gesture(self, gesture_name):
        if gesture_name == self.current_gesture:
            return
        self.current_gesture = gesture_name
        # TODO

    def stop(self):
        self.sport_client.Move(0, 0, 0)