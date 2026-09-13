#== Imports ====================================================
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from abstract_controller import AbstractGestureController

G1_NUM_MOTOR = 29                                   # number of actuators/motors a g1 robot has

# Torque = Kp * (target_angle - current_angle) + Kd * (target_speed - current_speed)
Kp = [
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40
]
Kd = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1, 
    1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1
]

GESTURE_TARGETS = {
    "turn_right":               {12:   -0.5},
    "turn_left":                {12:    0.5},
    "move_right":               {13:   -0.4},
    "move_left":                {13:    0.4},
    "move_forward_left_hand":   {15:    0.6},
    "move_forward_right_hand":  {22:    0.6},
    "move_backward_left_hand":  {15:   -0.6},
    "move_backward_right_hand": {22:   -0.6}
}

class Mode:
    PR = 0

class SimController(AbstractGestureController):
    def __init__(self):
        self.time_ = 0.0                            # self-timed clock
        self.control_dt_ = 0.002                    # time between sending messages to simulation 500/s
        self.ramp_duration_ = 3.0                   # time for 'ease into position' phase
        self.mode_machine_ = 0
        self.low_state = None                       # location of feedback data
        self.ready_ = False                         # check for if object has received their first bit of feedback data
        self.current_gesture = None                 # current gesture storage
        self.crc = CRC()                            # checksum calculator
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()# actual messaging object

    def init(self):
        ChannelFactoryInitialize(1, "lo")           # mujoco simulation: loopback
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)         # Broadcast object for LowCmd_ messages to "rt/lowcmd"
        self.lowcmd_publisher_.Init()                                           # Open broadcast connection
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)  # Listener object for LowState_ messages from "rt/lowstate"
        self.lowstate_subscriber.Init(self._on_low_state, 10)                   # Start Listener using a callback method for when a reply shows up and a message buffer of 10 messages

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        if not self.ready_:
            self.mode_machine_ = self.low_state.mode_machine
            self.ready_ = True

    def start(self):
        while not self.ready_:                      # waiting loop till self.ready_ = True (aka. gets a message)
            time.sleep(0.1)
        self.thread_ = RecurrentThread(interval=self.control_dt_, target=self._write, name="control")   # Configure reply
        self.thread_.Start()                        # Start replies

    def set_gesture(self, gesture_name):
        self.current_gesture = gesture_name

    def stop(self):
        pass  # TODO: graceful shutdown if needed

    def _write(self):
        self.time_ += self.control_dt_
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        if self.time_ < self.ramp_duration_:
            ratio = np.clip(self.time_ / self.ramp_duration_, 0.0, 1.0)
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]
        else:
            targets = GESTURE_TARGETS.get(self.current_gesture, {})
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = targets.get(i, 0.0)
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]
            # TODO: apply self.current_gesture's real joint trajectory here

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)