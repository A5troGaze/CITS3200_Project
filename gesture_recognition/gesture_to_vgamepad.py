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
import threading
import time

import vgamepad as vg

from abstract_controller import AbstractGestureController

# Gesture -> (vx, vy, wz). Same semantic units/signs as the rest of the project.
GESTURE_CMD = {
    # Turning is now full-deflection (+/-1.0) instead of a partial tilt.
    # Tried 0.2, then 0.4, then (independently, on Christo's end) 0.75 --
    # all three tested and none produced a real turn: wz is a joystick-tilt
    # FRACTION, and the actual ang_vel_z sent to the policy is that
    # fraction times deploy.yaml's trained turning range (only +/-0.2
    # rad/s -- much smaller than lin_vel_x's range for forward/back), so
    # anything less than full deflection was producing too small a real
    # angular velocity. Client suggested driving turning from a button
    # instead of the stick; going with the safer version of that idea --
    # treat the stick as on/off (always full deflection when the gesture
    # is active, zero otherwise) rather than binding a real controller
    # button, since we haven't confirmed g1_ctrl/unitree_mujoco actually
    # maps any physical button to a turn command, whereas the stick path
    # is already proven working for every other gesture. If even full
    # deflection doesn't produce a visible turn, the policy itself likely
    # isn't engaging its (slow) trained turning behaviour in this sim, and
    # that's a separate problem from stick magnitude -- flag back to the
    # team rather than continuing to tune this number further.
    "turn_right":                (0.0,  0.0, -1.0),
    "turn_left":                 (0.0,  0.0,  1.0),
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


class GestureGamepadBridge(AbstractGestureController):
    def __init__(self):
        self.gamepad = None
        self.current_gesture = None
        self._stopped = False
        self._thread = None

    def init(self):
        # Creating the virtual gamepad talks to the OS driver (ViGEmBus /
        # uinput), so do it here rather than in __init__, matching where
        # the other controllers open their DDS channel.
        self.gamepad = vg.VX360Gamepad()

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

    def start(self):
        # Blocks on the interactive stand-up/band-release sequence first
        # (same pattern as SimController.start() blocking until ready_),
        # then hands off to a background thread so the caller's own loop
        # (gesture_control.py's webcam loop) is free to keep calling
        # set_gesture() at its own pace.
        self.startup_sequence()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Neutral stick position so the robot doesn't keep walking on the
        # last-held command after we've stopped issuing gestures.
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.update()

    def _run_loop(self, rate_hz=50):
        while not self._stopped:
            vx, vy, wz = GESTURE_CMD.get(self.current_gesture, (0.0, 0.0, 0.0))
            self.gamepad.left_joystick_float(x_value_float=-vy, y_value_float=-vx)
            self.gamepad.right_joystick_float(x_value_float=-wz, y_value_float=0.0)
            self.gamepad.update()
            time.sleep(1.0 / rate_hz)


if __name__ == "__main__":
    # Standalone test path (no camera/gesture_control.py needed): runs the
    # stand-up/band-release sequence, then lets you type gesture names by
    # hand to sanity-check the joystick mapping before wiring in the
    # camera pipeline.
    bridge = GestureGamepadBridge()
    bridge.init()
    bridge.start()
    print(f"Streaming gestures. Type one of {list(GESTURE_CMD)} + Enter to test, "
          "blank to go idle, Ctrl+C to quit.")
    try:
        while True:
            typed = input("> ").strip()
            bridge.set_gesture(typed if typed in GESTURE_CMD else None)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()