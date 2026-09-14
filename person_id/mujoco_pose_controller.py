"""CycloneDDS controller for sending conservative G1 arm targets to MuJoCo.

This prototype is intended for the Unitree MuJoCo simulator, not a real robot.
It commands six arm joints, holds every other joint at its startup position,
clamps targets to configured limits, and rate-limits movement.
"""

import math
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


G1_NUM_MOTOR = 29  # Full G1 motor count; only six arm joints are targeted.

# Motor indices for the six joints controlled by this prototype.
LEFT_SHOULDER_PITCH = 15
LEFT_SHOULDER_ROLL = 16
LEFT_ELBOW = 18
RIGHT_SHOULDER_PITCH = 22
RIGHT_SHOULDER_ROLL = 23
RIGHT_ELBOW = 25

CONTROLLED_JOINTS = {
    LEFT_SHOULDER_PITCH,
    LEFT_SHOULDER_ROLL,
    LEFT_ELBOW,
    RIGHT_SHOULDER_PITCH,
    RIGHT_SHOULDER_ROLL,
    RIGHT_ELBOW,
}

# Position limits from the G1 29-DOF joint reference, in radians.
POSITION_LIMITS = {
    LEFT_SHOULDER_PITCH: (-3.0892, 2.6704),
    LEFT_SHOULDER_ROLL: (-1.5882, 2.2515),
    LEFT_ELBOW: (-1.0472, 2.0944),
    RIGHT_SHOULDER_PITCH: (-3.0892, 2.6704),
    RIGHT_SHOULDER_ROLL: (-2.2515, 1.5882),
    RIGHT_ELBOW: (-1.0472, 2.0944),
}

KP = [
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
]
KD = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
]


class Mode:
    PR = 0


def _clamp(index, value):
    # Reject unsupported joints and non-finite values before publishing to DDS.
    if index not in POSITION_LIMITS:
        raise ValueError(f"Motor {index} is not an allowed upper-body target")
    if not math.isfinite(value):
        raise ValueError(f"Motor {index} received a non-finite target")
    lower, upper = POSITION_LIMITS[index]
    return min(max(float(value), lower), upper)


class MujocoPoseController:
    """Publish selected upper-body joint targets while holding all other joints."""

    def __init__(
        self,
        domain_id=1,
        interface="lo",
        control_hz=200.0,
        max_command_speed=1.0,
    ):
        if control_hz <= 0:
            raise ValueError("control_hz must be greater than zero")
        if max_command_speed <= 0:
            raise ValueError("max_command_speed must be greater than zero")
        self.domain_id = domain_id
        self.interface = interface
        self.control_dt = 1.0 / control_hz
        self.max_command_step = float(max_command_speed) * self.control_dt
        self.low_state = None
        self.mode_machine = 0
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.targets = {}
        self.hold_positions = None  # First received position for all 29 joints.
        self.command_positions = None  # Published positions as they ramp to targets.
        self.thread = None
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

    def init(self, timeout=10.0):
        ChannelFactoryInitialize(self.domain_id, self.interface)
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._on_low_state, 10)

        # Fail clearly instead of waiting forever when MuJoCo is not publishing.
        if not self.ready.wait(timeout):
            raise TimeoutError(
                "No rt/lowstate received. Check that unitree_mujoco is running "
                f"with DOMAIN_ID={self.domain_id} and INTERFACE='{self.interface}'."
            )

    def _on_low_state(self, msg):
        self.low_state = msg
        if not self.ready.is_set():
            # Capture the first simulator state as the safe startup/return pose.
            self.mode_machine = msg.mode_machine
            self.hold_positions = [msg.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            self.command_positions = list(self.hold_positions)
            self.ready.set()

    def start(self):
        if not self.ready.is_set():
            raise RuntimeError("Call init() before start()")
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="mujoco-pose-control", daemon=True)
        self.thread.start()

    def set_targets(self, targets):
        """Set a partial {motor_index: angle_rad} upper-body target mapping."""
        cleaned = {int(index): _clamp(int(index), value) for index, value in targets.items()}
        with self.lock:
            self.targets = cleaned

    def clear_targets(self):
        """Return controlled joints to the positions captured during startup."""
        with self.lock:
            self.targets = {}

    def return_to_start(self, timeout=5.0, tolerance=0.02):
        """Rate-limit controlled joints back to their captured startup pose."""
        self.clear_targets()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                positions = list(self.command_positions)
            # Uncontrolled joints are omitted because they never leave startup.
            if all(
                abs(positions[index] - self.hold_positions[index]) <= tolerance
                for index in CONTROLLED_JOINTS
            ):
                return True
            time.sleep(min(0.02, self.control_dt))
        return False

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run(self):
        # Fixed-rate loop compensating for the time spent constructing each command.
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            self._write_once()
            next_tick += self.control_dt
            self.stop_event.wait(max(0.0, next_tick - time.monotonic()))

    def _write_once(self):
        with self.lock:
            targets = dict(self.targets)

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine

        for index in range(G1_NUM_MOTOR):
            # Unspecified joints continuously hold their captured startup position.
            desired = targets.get(index, self.hold_positions[index])
            current = self.command_positions[index]
            # Limit each step to avoid an instantaneous jump between video frames.
            delta = min(max(desired - current, -self.max_command_step), self.max_command_step)
            with self.lock:
                self.command_positions[index] = current + delta

            motor = self.low_cmd.motor_cmd[index]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = current + delta
            motor.dq = 0.0
            motor.kp = KP[index]
            motor.kd = KD[index]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
