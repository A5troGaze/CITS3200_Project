import vgamepad as vg

gp = vg.VX360Gamepad()
input("Virtual gamepad created. In another terminal run: ls /dev/input/js*  "
      "-- note which device it is, start raw_joystick_probe.py on that device, "
      "then come back and press Enter here to send L2 + Up...")

gp.left_trigger_float(value_float=1.0)
gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
gp.update()
input("L2 + Up is now asserted and held -- check the probe output, then press Enter to release...")

gp.left_trigger_float(value_float=0.0)
gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
gp.update()
print("Released.")
