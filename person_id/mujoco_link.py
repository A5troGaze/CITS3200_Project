"""
MuJoCo / Unitree SDK2 bridge.

Keeps Dhava's original LowState_ subscriber (rt/lowstate connection,
update_pose() hand-off point) and adds a rt/lowcmd publisher on top of it,
modelled on unitree_sdk2py's own
example/g1/low_level/g1_low_level_example.py: ChannelPublisher("rt/lowcmd",
LowCmd_), mode_pr=0 (PR mode), mode_machine copied from the latest
LowState_, motor_cmd[i].mode=1, dq=0, tau=0, and cmd.crc = CRC().Crc(cmd)
before every publish.

Publishes at a fixed 500 Hz (like that same example), holding the latest
retargeted target between vision updates and linearly interpolating from
the last published value toward it each tick, since vision runs at ~30 fps
and the command loop must not step. Refuses to publish anything until at
least one LowState_ has arrived.

KP/KD gains: COULD NOT be verified in this environment. unitree_sdk2_python
is an external dependency per SETUP.md, deliberately excluded from this
repo/checkout, and not reachable from this machine (the sim runs in a
separate Ubuntu VM). Per the task brief's own instruction not to invent
gain values, KP/KD are left as None below rather than guessed -- copy the
exact arrays (G1JointIndex order) from your local unitree_sdk2_python/
example/g1/low_level/g1_low_level_example.py before publishing to the sim
or hardware. Flagged again in the task's final summary.

Legs and waist (indices 0-14) are never retargeted -- the sim runs full
physics, so run this only with unitree_mujoco's elastic band enabled
(config enable_elastic_band: 1, toggled in the sim window) for arm-only
mirroring; otherwise an uncommanded/held lower body will make the robot
fall. See RUNNING_leader_pose.md.
"""

import threading
import time

import numpy as np

from g1_joint_limits import G1_29DOF_JOINT_LIMITS

PUBLISH_RATE_HZ = 500.0

# Kp/Kd, G1JointIndex order (indices 0-28). UNVERIFIED in this environment
# -- see the module docstring. Left as None so a missing fill-in fails
# loudly (see _publisher_loop) instead of silently commanding wrong gains.
KP = None
KD = None


class MujocoLink:
    def __init__(self, domain_id=1, interface="lo", dry_run=False):
        self.dry_run = dry_run
        self.latest_state = None
        self.latest_landmarks = None
        self.latest_frame_id = None
        self.latest_timestamp_ms = None
        self.last_print_time = 0.0

        self._target_q = None       # (29,) latest retargeted target
        self._published_q = None    # (29,) last value actually published
        self._lock = threading.Lock()
        self._publisher_thread = None
        self._stop = threading.Event()

        if dry_run:
            print("MujocoLink: --dry-run, no DDS channels opened")
            return

        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._default_lowcmd = unitree_hg_msg_dds__LowCmd_
        self._crc = CRC()

        ChannelFactoryInitialize(domain_id, interface)

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        print(f"MuJoCo link started on domain={domain_id}, interface='{interface}'")

    def _low_state_handler(self, msg):
        self.latest_state = msg

    def connected(self):
        return self.latest_state is not None

    def latest_lowstate_q(self):
        """(29,) array of the real LowState_.motor_state[i].q, in
        G1JointIndex order, or None if nothing has arrived yet (or in
        --dry-run, where there is no subscriber at all)."""
        if self.latest_state is None:
            return None
        return np.array([self.latest_state.motor_state[i].q for i in range(29)])

    def update_pose(self, frame_id, timestamp_ms, world_landmarks):
        """Kept for compatibility with live_pose_receiver.py and Dhava's
        original hand-off. Stores the latest raw landmarks only -- this
        does NOT retarget them. leader_pose.py --mujoco calls
        publish_target() with an already-retargeted RetargetResult instead;
        prefer that path for anything driving the robot."""
        self.latest_frame_id = frame_id
        self.latest_timestamp_ms = timestamp_ms
        self.latest_landmarks = world_landmarks

        now = time.time()
        if now - self.last_print_time >= 1.0:
            status = "connected" if self.connected() else "waiting for MuJoCo"
            print(f"Pose frame {frame_id} received; landmarks={len(world_landmarks)}; MuJoCo={status}")
            self.last_print_time = now

    def publish_target(self, retarget_result):
        """Update the target the 500 Hz publisher thread interpolates
        toward, starting that thread on first call. In --dry-run, prints
        the target instead (no DDS)."""
        if self.dry_run:
            print(f"[dry-run] frame {retarget_result.frame_id}: {retarget_result.q_array.tolist()}")
            return

        with self._lock:
            self._target_q = retarget_result.q_array.copy()

        if self._publisher_thread is None:
            self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
            self._publisher_thread.start()

    def _publisher_loop(self):
        if KP is None or KD is None:
            raise SystemExit(
                "mujoco_link.KP/KD are not filled in. Copy the exact Kp/Kd arrays "
                "(G1JointIndex order) from your local unitree_sdk2_python/example/"
                "g1/low_level/g1_low_level_example.py into person_id/mujoco_link.py "
                "before publishing to the sim or hardware -- these could not be "
                "verified from this environment. See the task's final summary."
            )

        dt = 1.0 / PUBLISH_RATE_HZ
        velocity_limits = np.array([lim.velocity for lim in G1_29DOF_JOINT_LIMITS.values()])
        max_step_per_tick = velocity_limits * dt

        while not self._stop.is_set():
            loop_start = time.time()

            if self.latest_state is None:
                # Refuse to publish until at least one LowState_ has arrived.
                time.sleep(dt)
                continue

            with self._lock:
                target = self._target_q

            if target is not None:
                if self._published_q is None:
                    self._published_q = np.array(
                        [self.latest_state.motor_state[i].q for i in range(29)]
                    )
                delta = np.clip(target - self._published_q, -max_step_per_tick, max_step_per_tick)
                self._published_q = self._published_q + delta

                cmd = self._default_lowcmd()
                cmd.mode_pr = 0
                cmd.mode_machine = self.latest_state.mode_machine
                for i in range(29):
                    cmd.motor_cmd[i].mode = 1
                    cmd.motor_cmd[i].q = float(self._published_q[i])
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].tau = 0.0
                    cmd.motor_cmd[i].kp = KP[i]
                    cmd.motor_cmd[i].kd = KD[i]
                cmd.crc = self._crc.Crc(cmd)
                self.lowcmd_publisher.Write(cmd)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    def close(self):
        self._stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)
        print("MuJoCo link closed")
