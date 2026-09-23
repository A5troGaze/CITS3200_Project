"""
gesture_to_vgamepad.py

Drives unitree_mujoco's existing joystick-reading code (pygame-based,
already correct/tested) via a virtual Xbox360 gamepad, instead of
publishing DDS messages ourselves. unitree_mujoco reads this virtual
gamepad and does the correct WirelessController_ + wireless_remote
byte-packing for us.

Axis/button map confirmed from unitree_mujoco's unitree_sdk2py_bridge.py
(js_type="xbox"): LX=axis0, LY=axis1, RX=axis3, RY=axis4, LT=axis2,
RT=axis5, A=btn0, B=btn1, X=btn2, Y=btn3, LB=btn4, RB=btn5,
SELECT=btn6, START=btn7, dpad via hat.

Velocity sign derivation (traced through both unitree_mujoco's bridge
AND g1_ctrl's observations.h):
  lin_vel_x = -left_stick_y   -> left_stick_y  = -vx
  lin_vel_y = -left_stick_x   -> left_stick_x  = -vy
  ang_vel_z = -right_stick_x  -> right_stick_x = -wz
"""

import subprocess
import time

import vgamepad as vg

# Gesture -> (vx, vy, wz). Same semantic units/signs as the rest of the project.
GESTURE_CMD = {
    "turn_right":                (0.0,  0.0, -0.2),
    "turn_left":                 (0.0,  0.0,  0.2),
    "move_right":                (0.0, -0.3,  0.0),
    "move_left":                 (0.0,  0.3,  0.0),
    "move_forward_left_hand":    (0.5,  0.0,  0.0),
    "move_forward_right_hand":   (0.5,  0.0,  0.0),
    "move_backward_left_hand":  (-0.5,  0.0,  0.0),
    "move_backward_right_hand": (-0.5,  0.0,  0.0),
}

STAND_SETTLE_DELAY = 2.0
GROUND_CONTACT_DELAY = 1.0
BAND_RELEASE_DELAY = 3.0
MUJOCO_WINDOW_NAME = "MuJoCo"  # -- VERIFY against your actual window title


def send_key_to_mujoco(key: str, times: int = 1, interval: float = 0.05):
    """Sends `key` to the MuJoCo window `times` times, with a short pause
    between presses so each one registers as a separate GLFW_PRESS event
    (each press of '8' only loosens the band by 0.1m -- see ElasticBand::length_
    in unitree_mujoco's main.cc)."""
    for _ in range(times):
        subprocess.run(
            ["xdotool", "search", "--name", MUJOCO_WINDOW_NAME, "key", "--window", "%1", key],
            check=False,
        )
        time.sleep(interval)


class GestureGamepadBridge:
    def __init__(self):
        self.gamepad = vg.VX360Gamepad()
        self.current_gesture = None

    def _pulse(self, press_fn, release_fn, hold_seconds=1.5):
        press_fn()
        self.gamepad.update()
        time.sleep(hold_seconds)
        release_fn()
        self.gamepad.update()

    def startup_sequence(self):
        input("Virtual gamepad is live. Start unitree_mujoco, then g1_ctrl "
            "in their own terminals, then press Enter here to begin...")

        print("Standing up (L2 + Up)...")
        # LT is a smoothed Axis (see unitree_joystick.hpp), Up is an instant Button.
        # Pressing both at once makes Up's on_pressed edge fire before LT finishes
        # ramping up, so "LT + up.on_pressed" never lines up. Hold LT alone first.
        self.gamepad.left_trigger_float(value_float=1.0)
        self.gamepad.update()
        time.sleep(2.0)  # let LT's smoothed value cross threshold on both ends
        self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.gamepad.update()
        time.sleep(1.0)  # hold the combo briefly so g1_ctrl definitely samples it
        self.gamepad.left_trigger_float(value_float=0.0)
        self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self.gamepad.update()
        input("Watch the MuJoCo window. Once the robot has risen into its FixStand "
            "pose and stopped moving, press Enter to ground its feet...")

        print("Grounding feet (viewer key 8)...")
        while True:
            send_key_to_mujoco("8", times=5)  # +0.5m of slack per batch
            answer = input("Are the robot's feet now touching the ground? "
                            "[Enter = not yet, loosen more] [d = done, feet are down]: ")
            if answer.strip().lower() == "d":
                break
        input("Press Enter once you're ready to start the walking policy (R1 + X)...")

        print("Running policy (R1 + X)...")
        self._pulse(
            lambda: (self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER), self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)),
            lambda: (self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER), self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)),
        )
        input("Watch the robot closely -- it should look actively balancing (small "
            "continuous joint motion), not limp or static. Once it looks stable "
            "and alive, press Enter to release the elastic band...")

        print("Releasing band (viewer key 9)...")
        send_key_to_mujoco("9")
        print("Startup sequence done -- streaming gesture commands.")

    def set_gesture(self, gesture_name):
        self.current_gesture = gesture_name

    def run_forever(self, rate_hz=50):
        while True:
            vx, vy, wz = GESTURE_CMD.get(self.current_gesture, (0.0, 0.0, 0.0))
            self.gamepad.left_joystick_float(x_value_float=-vy, y_value_float=-vx)
            self.gamepad.right_joystick_float(x_value_float=-wz, y_value_float=0.0)
            self.gamepad.update()
            time.sleep(1.0 / rate_hz)


if __name__ == "__main__":
    bridge = GestureGamepadBridge()
    bridge.startup_sequence()
    bridge.run_forever()