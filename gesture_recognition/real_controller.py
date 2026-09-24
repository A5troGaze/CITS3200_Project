from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from abstract_controller import AbstractGestureController

# Each gesture maps to a LocoClient.Move(vx, vy, vyaw) call:
#   vx   = forward(+) / backward(-) speed
#   vy   = left(+) / right(-) strafe speed
#   vyaw = turn left(+) / turn right(-) speed
# Sign conventions match SimController's GESTURE_TARGETS (negative = right).
# The real robot has no per-hand distinction while walking, so both
# "left hand" and "right hand" forward/backward gestures collapse onto the
# same forward/backward command.
GESTURE_ACTIONS = {
    "turn_right":               (0.0,  0.0, -0.3),
    "turn_left":                (0.0,  0.0,  0.3),
    "move_right":                (0.0, -0.3,  0.0),
    "move_left":                 (0.0,  0.3,  0.0),
    "move_forward_left_hand":   (0.3,  0.0,  0.0),
    "move_forward_right_hand":  (0.3,  0.0,  0.0),
    "move_backward_left_hand":  (-0.3, 0.0,  0.0),
    "move_backward_right_hand": (-0.3, 0.0,  0.0),
}


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

        action = GESTURE_ACTIONS.get(gesture_name)
        if action is None:
            # No gesture recognized (or an unmapped one) -> hold still
            # rather than keep repeating the last movement command.
            self.sport_client.Move(0, 0, 0)
            return

        vx, vy, vyaw = action
        self.sport_client.Move(vx, vy, vyaw)

    def stop(self):
        self.sport_client.Move(0, 0, 0)