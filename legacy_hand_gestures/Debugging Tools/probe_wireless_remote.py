import struct
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Matches BtnDataStruct in unitree_joystick.hpp:
#   uint8_t head[2]; BtnUnion btn (uint16_t); float lx, rx, ry, L2, ly;
FMT = "<2sH5f"
SIZE = struct.calcsize(FMT)

BIT_NAMES = ["R1", "L1", "Start", "Select", "R2", "L2", "f1", "f2",
             "A", "B", "X", "Y", "up", "right", "down", "left"]

last_printed = None

def handler(msg: LowState_):
    global last_printed
    raw = bytes(msg.wireless_remote[:SIZE])
    head, btn, lx, rx, ry, l2_axis, ly = struct.unpack(FMT, raw)
    pressed = [BIT_NAMES[i] for i in range(16) if btn & (1 << i)]
    line = f"btn_bits={pressed} L2_axis={l2_axis:.2f} lx={lx:.2f} ly={ly:.2f} rx={rx:.2f} ry={ry:.2f}"
    if line != last_printed:
        print(line)
        last_printed = line

ChannelFactoryInitialize(0, "lo")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(handler, 10)

print("Watching rt/lowstate wireless_remote decoded live (Ctrl+C to stop)...")
import time
while True:
    time.sleep(1)
