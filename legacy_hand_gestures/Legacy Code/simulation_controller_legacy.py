#== Imports ====================================================
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from abstract_controller import AbstractGestureController

G1_NUM_MOTOR = 29

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
        self.time_ = 0.0
        self.control_dt_ = 0.002
        self.ramp_duration_ = 3.0
        self.mode_machine_ = 0
        self.low_state = None
        self.ready_ = False
        self.current_gesture = None
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._stopped = False

    def init(self):
        ChannelFactoryInitialize(1, "lo")
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_low_state, 10)

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        if not self.ready_:
            self.mode_machine_ = self.low_state.mode_machine
            # Neutral hold target = the pose it's actually in right now.
            # Only valid if the robot is settled/band-supported when this fires,
            # not mid-fall.
            self.home_pose_ = [msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            self.ready_ = True

    def start(self):
        while not self.ready_:
            time.sleep(0.1)
        self.thread_ = RecurrentThread(interval=self.control_dt_, target=self._write, name="control")
        self.thread_.Start()

    def set_gesture(self, gesture_name):
        self.current_gesture = gesture_name

    def stop(self):
        self._stopped = True

    def _write(self):
        if self._stopped:
            return

        self.time_ += self.control_dt_
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        if self.time_ < self.ramp_duration_:
            ratio = np.clip(self.time_ / self.ramp_duration_, 0.0, 1.0)
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q + ratio * self.home_pose_[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]
        else:
            targets = GESTURE_TARGETS.get(self.current_gesture, {})
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = targets.get(i, self.home_pose_[i])
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].kd = Kd[i]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)