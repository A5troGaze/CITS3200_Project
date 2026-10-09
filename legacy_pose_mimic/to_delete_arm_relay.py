"""
arm_relay.py - sits between g1_ctrl and unitree_mujoco and overrides ONLY the arm joints.

    g1_ctrl (domain 0) --rt/lowcmd--> [relay: arms replaced by mimic targets] --rt/lowcmd--> unitree_mujoco (domain 1)
    g1_ctrl (domain 0) <--rt/lowstate-------------- [relay: pass-through] <----------------- unitree_mujoco (domain 1)

Mimic targets and an enable flag arrive from SimController as UDP/JSON on localhost:
    {"enable": true}   {"targets": {"15": 0.3, "16": -0.1, ...}}
When disabled (or when packets stop for STALE_S) the arms blend back to whatever g1_ctrl commands.

Run from this folder:  python arm_relay.py
"""

import json
import math
import socket
import threading
import time

from cyclonedds.core import Listener
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.qos import Policy, Qos
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from g1_joint_limits import G1_29DOF_JOINT_LIMITS
from rl_lab_policy import DEFAULT_POSE_MOTOR

CTRL_DOMAIN = 0          # g1_ctrl (hard-coded to domain 0)
SIM_DOMAIN = 1           # unitree_mujoco config.yaml -> domain_id: 1
INTERFACE = "lo"
UDP_ADDR = ("127.0.0.1", 9870)

OVERRIDE_JOINTS = tuple(range(15, 29))   # both arms. Add 12, 13, 14 for the waist.
ARM_KP = None            # None = keep the gains g1_ctrl sends. e.g. 40.0 to force your own.
ARM_KD = None            # e.g. 2.0

SMOOTH_TAU = 0.05
MAX_SPEED = 1.0          # arms overhead only stay up at <= 1 rad/s (teammate's finding)
RAMP_S = 0.5
STALE_S = 1.0

LIMITS = {lim.index: (lim.lower, lim.upper) for lim in G1_29DOF_JOINT_LIMITS.values()}


def make_participant(domain_id, keepalive):
    xml = ('<CycloneDDS><Domain Id="any"><General><Interfaces>'
           f'<NetworkInterface name="{INTERFACE}" priority="default" multicast="default"/>'
           '</Interfaces></General></Domain></CycloneDDS>')
    keepalive.append(Domain(domain_id, xml))
    return DomainParticipant(domain_id)


class Relay:
    def __init__(self):
        self._keep = []
        ctrl = make_participant(CTRL_DOMAIN, self._keep)
        sim = make_participant(SIM_DOMAIN, self._keep)
        self._crc = CRC()
        read_qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        self._cmd_out = DataWriter(sim, Topic(sim, "rt/lowcmd", LowCmd_))
        self._state_out = DataWriter(ctrl, Topic(ctrl, "rt/lowstate", LowState_))
        self._state_in = DataReader(sim, Topic(sim, "rt/lowstate", LowState_), qos=read_qos,
                                    listener=Listener(on_data_available=self._on_state))
        self._cmd_in = DataReader(ctrl, Topic(ctrl, "rt/lowcmd", LowCmd_), qos=read_qos,
                                  listener=Listener(on_data_available=self._on_cmd))
        self._lock = threading.Lock()
        self._enabled = False
        self._targets = {}
        self._last_packet = 0.0
        self._weight = 0.0
        self._shaped = None
        self._last_t = time.monotonic()
        self._n_state = 0
        self._n_cmd = 0

    def _on_state(self, reader):
        for msg in reader.take(N=8):
            w = self._weight                     # 0 = pass-through, 1 = fully overridden
            if w > 0.0:
                for j in OVERRIDE_JOINTS:        # tell g1_ctrl the arms are at default, not moving
                    s = msg.motor_state[j]
                    s.q = (1.0 - w) * s.q + w * float(DEFAULT_POSE_MOTOR[j])
                    s.dq = (1.0 - w) * s.dq
            self._state_out.write(msg)
            self._n_state += 1

    def _on_cmd(self, reader):
        for msg in reader.take(N=8):
            self._n_cmd += 1
            self._apply_override(msg)
            self._cmd_out.write(msg)

    def _apply_override(self, msg):
        now = time.monotonic()
        dt = min(max(now - self._last_t, 0.0005), 0.02)
        self._last_t = now
        with self._lock:
            enabled = self._enabled and (now - self._last_packet) < STALE_S
            targets = dict(self._targets)
        step = dt / RAMP_S
        self._weight = min(1.0, self._weight + step) if enabled else max(0.0, self._weight - step)
        if self._weight <= 0.0:
            self._shaped = None
            return
        if self._shaped is None:
            self._shaped = {j: msg.motor_cmd[j].q for j in OVERRIDE_JOINTS}
        w = self._weight
        for j in OVERRIDE_JOINTS:
            lo, hi = LIMITS[j]
            target = min(max(targets.get(j, self._shaped[j]), lo), hi)
            delta = (target - self._shaped[j]) * min(1.0, dt / SMOOTH_TAU)
            cap = MAX_SPEED * dt
            self._shaped[j] += max(-cap, min(cap, delta))
            m = msg.motor_cmd[j]
            m.q = w * self._shaped[j] + (1.0 - w) * m.q
            m.dq = 0.0
            if ARM_KP is not None:
                m.kp = ARM_KP
            if ARM_KD is not None:
                m.kd = ARM_KD
        msg.crc = self._crc.Crc(msg)

    def _udp_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(UDP_ADDR)
        while True:
            data, _ = sock.recvfrom(4096)
            try:
                packet = json.loads(data)
                with self._lock:
                    self._last_packet = time.monotonic()
                    if "enable" in packet:
                        new = bool(packet["enable"])
                        if new and not self._enabled:
                            self._targets.clear()    # fresh mimic session: forget old targets
                        self._enabled = new
                    for k, v in packet.get("targets", {}).items():
                        j, v = int(k), float(v)
                        if j in OVERRIDE_JOINTS and math.isfinite(v):
                            self._targets[j] = v
            except (ValueError, TypeError, AttributeError):
                continue

    def run(self):
        threading.Thread(target=self._udp_loop, daemon=True).start()
        print(f"[relay] domain {CTRL_DOMAIN} (g1_ctrl) <-> domain {SIM_DOMAIN} (sim) on '{INTERFACE}'. "
              f"UDP {UDP_ADDR}. Ctrl+C to stop.")
        t0, n_s0, n_c0 = time.monotonic(), 0, 0
        while True:
            time.sleep(5.0)
            now = time.monotonic()
            print(f"[relay] state {(self._n_state - n_s0) / (now - t0):.0f}/s  "
                  f"cmd {(self._n_cmd - n_c0) / (now - t0):.0f}/s  "
                  f"arm override weight {self._weight:.2f}")
            t0, n_s0, n_c0 = now, self._n_state, self._n_cmd


if __name__ == "__main__":
    Relay().run()