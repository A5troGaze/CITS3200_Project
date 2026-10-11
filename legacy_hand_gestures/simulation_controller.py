"""
simulation_controller.py
"""

#------------------------------------------------------------------------#
#   Imports
#------------------------------------------------------------------------#

import os
import subprocess
import sys
import threading
import time

import vgamepad as vg

from abstract_controller import AbstractGestureController


#------------------------------------------------------------------------#
#   Configuration
#------------------------------------------------------------------------#

MODE_GESTURE = "gesture"
MODE_MIMIC = "mimic"

MUJOCO_WINDOW_NAME = "MuJoCo"
#G1_CTRL_PROCESS = "g1_ctrl"

RATE_HZ = 50
MIMIC_SETTLE_S = 1.0    # Let the robot come to rest before the arms are taken over
#BAND_SETTLE_S = 1.0     # Let the band take hold before/after switching

ARM_RAMP_S = 1.0        # arm_sdk weight ramp 0 -> 1 / 1 -> 0 (publisher default is 2.0)
ARM_MAX_SPEED = 1.0     # rad/s: arms overhead only stay up at <= 1 rad/s (teammate's finding)

# Gesture -> x, y, r commands
GESTURE_CMD = {
    "move_right":                (0.0, -0.3,  0.0),
    "move_left":                 (0.0,  0.3,  0.0),
    "move_forward_left_hand":    (0.5,  0.0,  0.0),
    "move_forward_right_hand":   (0.5,  0.0,  0.0),
    "move_backward_left_hand":  (-0.5,  0.0,  0.0),
    "move_backward_right_hand": (-0.5,  0.0,  0.0),
}

# Turning -> three point turning logic
TURN_FORWARD_S = 1.5
TURN_BACKWARD_S = 1.5
TURN_GESTURES = {"turn_left", "turn_right"}

TURN_FORWARD_CMD = {
    "turn_left":  (0.3, 0.0, 0.7),
    "turn_right": (0.3, 0.0, -0.7),
}

TURN_BACKWARD_CMD = {
    "turn_left":  (-0.3, 0.0, 0.7),
    "turn_right": (-0.3, 0.0, -0.7),
}


#------------------------------------------------------------------------#
#   Helpers
#------------------------------------------------------------------------#

def send_key_to_mujoco(key: str, times: int = 1, interval: float = 0.05):
    """Sends `key` to the unitree_mujoco viewer window. (needs xdotool)."""
    for _ in range(times):
        subprocess.run(
            ["xdotool", "search", "--name", MUJOCO_WINDOW_NAME, "key", "--window", "%1", key],
            check=False,
        )
        time.sleep(interval)


#------------------------------------------------------------------------#
#   Controller
#------------------------------------------------------------------------#

class SimController(AbstractGestureController):
    def __init__(self):
        self.gamepad = None
        self.mode = MODE_GESTURE
        self.current_gesture = None

        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None

        self._turn_phase_start = 0.0
        self._turn_going_forward = True
        self._arms = None               # ArmSdkPublisher (rt/arm_sdk)
        self._mimic_active = False


    #--------------------------------------------------------------------#
    #   AbstractGestureController interface
    #--------------------------------------------------------------------#

    def init(self):
        self.gamepad = vg.VX360Gamepad()
        self._neutral()

    def start(self):
        self.startup_sequence()
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._stop_mimic()
        finally:
            if self._arms is not None:
                self._arms.stop()
            self._neutral()

    def set_gesture(self, gesture_name):
        with self._lock:
            if self.mode != MODE_GESTURE:
                return
            if gesture_name in TURN_GESTURES and gesture_name != self.current_gesture:
                self._turn_phase_start = time.monotonic()
                self._turn_going_forward = True
            self.current_gesture = gesture_name

    def on_mode_change(self, mode):
        if mode == self.mode:
            return

        if mode == MODE_MIMIC:
            with self._lock:
                self.current_gesture = None
                self.mode = MODE_MIMIC
            try:
                time.sleep(MIMIC_SETTLE_S)      # sticks are neutral; let the robot settle
                self._prepare_mimic()
                self._arms.engage()             # weight ramps 0 -> 1; g1_ctrl keeps balancing
                self._mimic_active = True
            except Exception:
                self._stop_mimic()
                with self._lock:
                    self.mode = MODE_GESTURE
                raise

        elif mode == MODE_GESTURE:
            self._stop_mimic()              # arms ease home, weight ramps 1 -> 0
            with self._lock:
                self.mode = MODE_GESTURE

        else:
            raise ValueError(f"UNKNOWN MODE: {mode!r}")


    #--------------------------------------------------------------------#
    #   Mimic: arm targets go out on rt/arm_sdk, g1_ctrl keeps balancing
    #--------------------------------------------------------------------#

    def set_joint_targets(self, targets):
        '''Arm joint targets from the pose pipeline. Ignored outside mimic mode.'''
        if self.mode == MODE_MIMIC and self._mimic_active:
            try:
                self._arms.set_targets(targets)
            except ValueError as e:
                print(f"[Mimic] Dropped targets: {e}")

    def _prepare_mimic(self):
        '''One-time setup: DDS connect to the simulator, start the 50 Hz publisher thread.'''
        if self._arms is None:
            person_id_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "person_id"))
            if person_id_dir not in sys.path:
                sys.path.append(person_id_dir)
            from arm_sdk_publisher import ArmSdkPublisher
            # real=True only means "actually publish". interface="lo" = the simulator, never a robot.
            arms = ArmSdkPublisher(interface="lo", domain_id=0, real=True,
                                   ramp_s=ARM_RAMP_S, max_command_speed=ARM_MAX_SPEED)
            arms.init()      # waits for rt/lowstate, records the standing arm pose as the start pose
            arms.start()     # publishes weight 0 until engage()
            self._arms = arms

    def _stop_mimic(self):
        if not self._mimic_active:
            return
        try:
            if not self._arms.disengage():
                print("[Mimic] arm_sdk weight did not reach 0 in time")
        finally:
            self._mimic_active = False


    #--------------------------------------------------------------------#
    #   Startup: Lower robot -> FixStand -> Walking Policy -> Release band
    #--------------------------------------------------------------------#

    def startup_sequence(self):
        input("Virtual gamepad is live.\nStart unitree_mujoco, then g1_ctrl in their own terminals.\nThen press Enter here to begin...")

        # Put robot into the starting pose
        print("[Startup] FixStand (L2 + Up)...")
        self.gamepad.left_trigger_float(value_float=1.0)
        self.gamepad.update()
        time.sleep(2.0)
        self._pulse(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP, hold=1.0)
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.update()

        # Lower robot to ground
        while True:
            send_key_to_mujoco("8", times=5)  # +0.5m of slack per batch
            if input("Are the robot's feet on the ground?\n[Enter = not yet, loosen more] [y = yes]: ").strip().lower() == "y":
                break

        # Start walking policy
        print("[Startup] Starting walking policy (R1 + X)...")
        self._pulse(
            vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
            vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            hold=0.3
        )
        time.sleep(1.0)

        # Release elastic band
        print("[Startup] Releasing elastic band (key 9)...")
        send_key_to_mujoco("9")
        '''
        self._band_on = False
        '''
        time.sleep(1.0)
        print("[Startup] Sequence done: Streaming gesture commands")


    #--------------------------------------------------------------------#
    #   Gamepad output
    #--------------------------------------------------------------------#

    def _current_velocity(self):
        '''Return (vx, vy, wz) for the current gesture.'''
        with self._lock:
            if self.mode != MODE_GESTURE:
                return 0.0, 0.0, 0.0
            gesture = self.current_gesture

            if gesture in TURN_GESTURES:
                now = time.monotonic()
                phase_limit = TURN_FORWARD_S if self._turn_going_forward else TURN_BACKWARD_S
                if now - self._turn_phase_start >= phase_limit:
                    self._turn_going_forward = not self._turn_going_forward
                    self._turn_phase_start = now

                cmd_table = TURN_FORWARD_CMD if self._turn_going_forward else TURN_BACKWARD_CMD
                return cmd_table[gesture]

            return GESTURE_CMD.get(gesture, (0.0, 0.0, 0.0))

    def _run_loop(self):
        period = 1.0 / RATE_HZ
        while not self._stopped.is_set():
            vx, vy, wz = self._current_velocity()
            self.gamepad.left_joystick_float(x_value_float=-vy, y_value_float=-vx)
            self.gamepad.right_joystick_float(x_value_float=-wz, y_value_float=0.0)
            self.gamepad.update()
            time.sleep(period)


    #--------------------------------------------------------------------#
    #   Low-level gamepad helpers
    #--------------------------------------------------------------------#

    def _neutral(self):
        if self.gamepad is None:
            return
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.update()

    def _pulse(self, *buttons, hold=0.2):
        for b in buttons:
            self.gamepad.press_button(b)
        self.gamepad.update()
        time.sleep(hold)
        for b in buttons:
            self.gamepad.release_button(b)
        self.gamepad.update()


#------------------------------------------------------------------------#
#   Manual test
#------------------------------------------------------------------------#

if __name__ == "__main__":
    controller = SimController()
    controller.init()
    controller.start()

    try:
        for g in ["move_right", "move_left", "move_forward_left_hand", "move_forward_right_hand",
                  "move_backward_left_hand", "move_backward_right_hand", "turn_left", "turn_right", None]:
            print(f"GESTURE: {g}")
            controller.set_gesture(g)
            time.sleep(4.0)

        print("MODE: Mimic (arms on rt/arm_sdk, g1_ctrl still balancing)")
        controller.on_mode_change(MODE_MIMIC)
        time.sleep(3.0)

        print("MODE: Gesture (arms handed back to g1_ctrl)")
        controller.on_mode_change(MODE_GESTURE)
        time.sleep(3.0)

    finally:
        controller.stop()