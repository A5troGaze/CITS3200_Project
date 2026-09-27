from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

def handler(msg: LowState_):
    print("Got LowState_! tick =", msg.tick())

ChannelFactoryInitialize(0, "lo")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(handler, 10)

print("Listening on rt/lowstate, domain 0, interface lo... (Ctrl+C to stop)")
import time
while True:
    time.sleep(1)
