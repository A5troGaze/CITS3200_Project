"""
MuJoCo / Unitree SDK2 connection helper.

This module is intentionally small:
- It connects to the running Unitree MuJoCo simulator.
- It listens for LowState messages from the simulated G1.
- It receives MediaPipe world landmarks from pose_test-world_landmarks.py.

It does NOT control the robot yet. Robot joint mapping/control belongs to the
DOF / retargeting task, not this connection task.
"""

import time


class MujocoLink:
    def __init__(self, domain_id=1, interface="lo"):
        self.latest_state = None
        self.latest_landmarks = None
        self.latest_frame_id = None
        self.latest_timestamp_ms = None
        self.last_print_time = 0.0

        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        ChannelFactoryInitialize(domain_id, interface)

        self.lowstate_subscriber = ChannelSubscriber(
            "rt/lowstate",
            LowState_,
        )

        self.lowstate_subscriber.Init(
            self._low_state_handler,
            10,
        )

        print(
            f"MuJoCo link started on domain={domain_id}, interface='{interface}'"
        )

    def _low_state_handler(self, msg):
        self.latest_state = msg

    def connected(self):
        return self.latest_state is not None

    def update_pose(self, frame_id, timestamp_ms, world_landmarks):
        """
        Receive the latest MediaPipe world landmarks.

        world_landmarks should be MediaPipe's 33 pose_world_landmarks.
        This is the hand-off point for later robot retargeting.
        """
        self.latest_frame_id = frame_id
        self.latest_timestamp_ms = timestamp_ms
        self.latest_landmarks = world_landmarks

        now = time.time()
        if now - self.last_print_time >= 1.0:
            status = "connected" if self.connected() else "waiting for MuJoCo"
            print(
                f"Pose frame {frame_id} received; "
                f"landmarks={len(world_landmarks)}; "
                f"MuJoCo={status}"
            )
            self.last_print_time = now

    def close(self):
        print("MuJoCo link closed")
