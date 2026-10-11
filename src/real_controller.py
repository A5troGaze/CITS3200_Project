'''
real_controller.py

Controller for the physical Unitree G1. Turns each recognised gesture into a
walking command and sends it to the robot over the network.

Currently gesture mode only. Mimic mode (copying the user's arms) is not
implemented yet. A commented-out, untested draft is included below, marked
with "MIMIC MODE DRAFT".
'''

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from abstract_controller import AbstractController

#-- MIMIC MODE DRAFT (1/4): import ------------------------------------------
# WARNING: NOT TESTED ON THE PHYSICAL ROBOT. Do not un-comment without a safe
# test setup (robot supported on a gantry/harness, someone holding the remote's
# emergency stop, speed cap kept low). See SETUP.md / README.md.
# from arm_sdk_publisher import ArmSdkPublisher


#----------------------------------------------------------
# Gesture -> walking command
#----------------------------------------------------------
# Each gesture maps to a LocoClient.Move(vx, vy, vyaw) call:
#   vx   = forward(+) / backward(-) speed
#   vy   = left(+) / right(-) strafe speed
#   vyaw = turn left(+) / turn right(-) speed
# Sign conventions are the same as in the simulation controller (negative = right).
# The real robot has no per-hand distinction while walking, so both
# "left hand" and "right hand" forward/backward gestures collapse onto the
# same forward/backward command.
GESTURE_ACTIONS = {
    "turn_right":               (0.0,  0.0, -0.3),
    "turn_left":                (0.0,  0.0,  0.3),
    "move_right":               (0.0, -0.3,  0.0),
    "move_left":                (0.0,  0.3,  0.0),
    "move_forward_left_hand":   (0.3,  0.0,  0.0),
    "move_forward_right_hand":  (0.3,  0.0,  0.0),
    "move_backward_left_hand":  (-0.3, 0.0,  0.0),
    "move_backward_right_hand": (-0.3, 0.0,  0.0),
}

#-- MIMIC MODE DRAFT (2/4): constants ---------------------------------------
# ARM_RAMP_S = 1.0        # seconds to hand the arms over (same as the simulator)
# ARM_MAX_SPEED = 1.0     # rad/s limit on arm movement (same as the simulator)


#----------------------------------------------------------
# Controller
#----------------------------------------------------------
class RealController(AbstractController):
    def __init__(self, interface="wlan0"):
        self.interface = interface      # network interface connected to the robot
        self.current_gesture = None     # last gesture sent (avoids resending the same command)

        #-- MIMIC MODE DRAFT (3/4): state ------------------------------------
        # self._arms = None               # created the first time mimic mode starts
        # self._mimic_active = False

    def init(self):
        '''Connect to the robot (DDS domain 0) and set up the walking client.'''
        ChannelFactoryInitialize(0, self.interface)
        self.sport_client = LocoClient()
        self.sport_client.SetTimeout(10.0)  # seconds to wait for the robot to answer a command
        self.sport_client.Init()

    def start(self):
        pass    # nothing to do: the robot is already ready to walk

    def set_gesture(self, gesture_name):
        '''Send the walking command for this gesture (only when the gesture changes).'''
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
        '''Stop walking.'''
        self.sport_client.Move(0, 0, 0)

    #-- MIMIC MODE DRAFT (4/4): methods ----------------------------------------
    # WARNING: NOT TESTED ON THE PHYSICAL ROBOT (see note at the top of this file).
    # If you un-comment this, also:
    #   - un-comment parts 1, 2 and 3 above
    #   - DELETE the active stop() above (the version below replaces it)
    #   - update the module docstring
    #
    # def on_mode_change(self, mode):
    #     '''Mimic mode: stop walking and give the arms to arm_sdk.
    #     Gesture mode: hand the arms back to the robot's own controller.'''
    #     if mode == "mimic":
    #         self.sport_client.Move(0, 0, 0)     # stand still first
    #         self.current_gesture = None         # so the next gesture is always sent
    #         if self._arms is None:
    #             self._arms = ArmSdkPublisher(interface=self.interface, domain_id=0, real=True,
    #                                          ramp_s=ARM_RAMP_S, max_command_speed=ARM_MAX_SPEED)
    #             self._arms.init()
    #             self._arms.start()
    #         self._arms.engage()
    #         self._mimic_active = True
    #     elif self._mimic_active:
    #         self._arms.disengage()
    #         self._mimic_active = False
    #
    # def set_joint_targets(self, targets):
    #     '''Mimic mode: pass the arm angles from the mimic source to the robot.'''
    #     if self._mimic_active:
    #         self._arms.set_targets(targets)
    #
    # def stop(self):
    #     '''Stop walking and hand the arms back before shutting down.'''
    #     self.sport_client.Move(0, 0, 0)
    #     if self._arms is not None:
    #         if self._mimic_active:
    #             self._arms.disengage()
    #             self._mimic_active = False
    #         self._arms.stop()