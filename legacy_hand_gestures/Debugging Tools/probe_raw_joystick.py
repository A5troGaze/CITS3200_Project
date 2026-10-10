import struct
import sys

JS_EVENT_FORMAT = "IhBB"  # time(4 bytes), value(2 bytes), type(1 byte), number(1 byte)
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80  # flag bit for the synthetic "initial state" events sent at open time

device = sys.argv[1] if len(sys.argv) > 1 else "/dev/input/js0"

with open(device, "rb") as f:
    print(f"Reading raw events from {device} (Ctrl+C to stop)...")
    while True:
        data = f.read(JS_EVENT_SIZE)
        if not data:
            break
        _time, value, type_, number = struct.unpack(JS_EVENT_FORMAT, data)
        base_type = type_ & ~JS_EVENT_INIT
        kind = "AXIS" if base_type == JS_EVENT_AXIS else "BUTTON" if base_type == JS_EVENT_BUTTON else "OTHER"
        tag = " (init)" if type_ & JS_EVENT_INIT else ""
        print(f"{kind}{tag} #{number} = {value}")
