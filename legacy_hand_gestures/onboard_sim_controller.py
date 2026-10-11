"""
onboard_sim_controller.py

Simulation backend where gesture mode AND mimic mode both work, on one
simulator: person_id's sim_standing.py --rl-lab.

That simulator runs unitree_rl_lab's walking policy inside the physics loop
(like Unitree's own locomotion on the real G1), so the robot is balanced the
whole time and nothing has to be paused when the mode changes:

  gesture mode: gesture -> (vx, vy, wz) on rt/cmd_vel -> the policy walks
                (same GESTURE_CMD table and turn logic as SimController)
  mimic mode:   walking command held at zero; PersonIdBrain's arm angles go
                out on rt/arm_sdk (person_id's ArmSdkPublisher), and the
                simulator blends them over the policy's arms. The legs stay
                with the policy, which keeps balancing.

Switching to mimic ramps the arm_sdk weight 0 -> 1 (the arms ease over to
the person); switching back eases the arms to where they started, then hands
them back to the policy.

Run (in the VM, g1-env active, from ~/CITS3200/Project):
    terminal 1:  cd person_id && python3 sim_standing.py --rl-lab
    terminal 2:  cd gesture_recognition && python3 gesture_control.py onboard
                 (once terminal 1 prints "Band released")

No g1_ctrl, unitree_mujoco or virtual gamepad is used by this backend.
"""
import os
import sys
import threading

from simulation_controller import SimController

PERSON_ID_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "person_id"))


class OnboardSimController(SimController):
    def __init__(self, domain_id=1, interface="lo", arm_speed=1.0):
        # domain 1 / lo: person_id's simulator settings (person_id/INSTALL.md step 8).
        # arm_speed 1.0 rad/s: faster arm moves can knock the balance policy over.
        super().__init__()
        self.domain_id = domain_id
        self.interface = interface
        self.arm_speed = arm_speed
        self.mode = "gesture"
        self.arms = None
        self.velocity_pub = None
        self._velocity_msg = None
        self._mode_lock = threading.Lock()
        self._stop_event = threading.Event()

    def init(self):
        if PERSON_ID_DIR not in sys.path:
            sys.path.append(PERSON_ID_DIR)      # person_id's modules use plain imports
        from arm_sdk_publisher import ArmSdkPublisher
        import velocity_cmd

        self.arms = ArmSdkPublisher(interface=self.interface, domain_id=self.domain_id, real=True,
                                    control_hz=100.0, max_command_speed=self.arm_speed)
        print("Waiting for rt/lowstate from the simulator "
              "(start it with: cd person_id && python3 sim_standing.py --rl-lab)...")
        self.arms.init()                        # also initialises the DDS channel factory

        from unitree_sdk2py.core.channel import ChannelPublisher

        self._velocity_msg = velocity_cmd.message_type()
        self.velocity_pub = ChannelPublisher(velocity_cmd.TOPIC, self._velocity_msg)
        self.velocity_pub.Init()

    def start(self):
        # No stand-up sequence: the simulator stands the robot up and releases
        # the band by itself. The arm publisher starts at weight 0, so the
        # policy keeps the arms until mimic mode engages them.
        self.arms.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("Onboard sim ready: gestures walk, 'm' switches to arm mimicry.")

    def _publish_velocity(self, vx, vy, wz):
        self.velocity_pub.Write(self._velocity_msg(vx=float(vx), vy=float(vy), vyaw=float(wz)))

    def _run_loop(self, rate_hz=50):
        # Published continuously (also zeros): the simulator stops the robot
        # if rt/cmd_vel goes quiet for 0.5 s, e.g. if this process dies.
        while not self._stopped:
            if self.mode == "gesture":
                self._publish_velocity(*self._current_velocity())
            else:
                self._publish_velocity(0.0, 0.0, 0.0)
            self._stop_event.wait(1.0 / rate_hz)

    def set_joint_targets(self, targets):
        if self.mode == "mimic" and targets:
            self.arms.set_targets(targets)

    def on_mode_change(self, mode):
        with self._mode_lock:
            if mode == "mimic":
                self.current_gesture = None
                self.mode = "mimic"             # stop walking first...
                self.arms.engage()              # ...then take the arms
            elif mode == "gesture" and self.mode == "mimic":
                print("Returning the arms to the balance policy...")
                self.arms.disengage()           # blocks a few seconds while the arms ease back
                self.mode = "gesture"

    def stop(self):
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.velocity_pub is not None:
            self._publish_velocity(0.0, 0.0, 0.0)
        if self.arms is not None:
            if self.mode == "mimic":
                self.arms.disengage()
            self.arms.stop()
