"""
unitree_mujoco with the G1 standing on the floor instead of hanging from
the elastic band.

The stock simulator (simulate_python/unitree_mujoco.py) has two options for
a robot that nobody is balancing: keep the elastic band on, so the G1 hangs
with its feet off the floor and swings whenever its arms move, or turn it off
(key 9 / ENABLE_ELASTIC_BAND = False) and watch it fall. We checked the
second option offline: holding the legs, even at 6x the SDK gains, the G1
falls within a few seconds of arm motion. The ankles are limited to
50 N*m, and balance needs an active controller (out of scope for person_id;
on the real robot Unitree's locomotion does it).

This launcher runs the same simulator, on the same scene, bridge, topics and
DDS settings, but loads the G1 with its floating base removed: the pelvis is
pinned at its standing height, the feet rest on the floor, and there is no band.
Everything above the pelvis (legs, waist, arms) moves exactly as in the
stock sim. Nothing in unitree_mujoco is modified; its bridge is imported
from simulate_python/.

    python3 sim_standing.py                 # instead of: python3 unitree_mujoco.py
    python3 sim_standing.py --viewer-fps 20 # lighter on a VM without a GPU

Then run leader_pose.py / person_id_replay.py / test_dds_fixed_target.py as usual.
The IMU reports a constant upright pelvis, as a balanced robot's roughly would.

It also publishes rt/lowstate at 50 Hz instead of 200 Hz (--state-hz): on
the VM the stock rate starves the physics thread (see simulate() below);
it prints its speed relative to real time every 10 s.
"""

import argparse
import os
import sys
import threading
import time

from g1_sim import UNITREE_MUJOCO, load_sim_model

SIM_DIR = os.path.join(UNITREE_MUJOCO, "simulate_python")


def main():
    sys.path.insert(0, SIM_DIR)
    import config  # unitree_mujoco's own simulate_python/config.py

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain-id", type=int, default=config.DOMAIN_ID)
    parser.add_argument("--interface", default=config.INTERFACE)
    parser.add_argument("--dt", type=float, default=0.002,
                        help="physics step (s). The MJCF's own 0.002; the stock launcher uses 0.005, where "
                             "the explicit PD on the light wrist joints is close to unstable")
    parser.add_argument("--viewer-fps", type=float, default=15.0,
                        help="viewer refresh rate; the VM renders in software, so lower is lighter")
    parser.add_argument("--state-hz", type=float, default=50.0,
                        help="rt/lowstate publish rate (stock sim: every physics step, 200 Hz)")
    parser.add_argument("--headless", action="store_true",
                        help="no viewer window (e.g. with --record-demo, which renders its own view)")
    args = parser.parse_args()
    if config.ROBOT != "g1":
        raise SystemExit(f"simulate_python/config.py has ROBOT = {config.ROBOT!r}; this launcher is for the g1")

    import mujoco
    import mujoco.viewer
    import numpy as np
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge

    class PerStepPdBridge(UnitreeSdk2Bridge):
        """unitree_mujoco's bridge, except a received LowCmd is stored rather
        than turned into a torque once (see apply_pd in simulate())."""

        def __init__(self, *a, **kw):
            self.latest = {"cmd": None}
            super().__init__(*a, **kw)

        def LowCmdHandler(self, msg):
            n = self.num_motor
            mc = msg.motor_cmd
            self.latest["cmd"] = (
                np.array([mc[i].q for i in range(n)]), np.array([mc[i].dq for i in range(n)]),
                np.array([mc[i].kp for i in range(n)]), np.array([mc[i].kd for i in range(n)]),
                np.array([mc[i].tau for i in range(n)]),
            )

    model = load_sim_model(fixed_base=True)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nu = model.nu
    lock = threading.Lock()
    viewer = None if args.headless else mujoco.viewer.launch_passive(model, data)
    running = (lambda: True) if viewer is None else viewer.is_running
    time.sleep(0.2)

    def simulate():
        ChannelFactoryInitialize(args.domain_id, args.interface)
        # The bridge publishes rt/lowstate (and sportmodestate) on threads whose
        # period is model.opt.timestep at construction. One Python LowState
        # publish costs ~4 ms on the team VM, so at the stock 200 Hz those
        # threads hold the GIL almost continuously and physics runs at ~0.1x
        # real time. Build the bridge with a longer period, then restore the
        # physics step.
        model.opt.timestep = 1.0 / args.state_hz
        bridge = PerStepPdBridge(model, data)
        model.opt.timestep = args.dt

        # The bridge computes the PD torque once per received LowCmd and then
        # holds it. When physics has to catch up (several steps between two
        # messages) that stale torque destabilises the joints, and after the
        # publisher stops it drives them onto their limits. Here the latest
        # LowCmd is kept and the bridge's own formula
        #   ctrl = tau + kp * (q_des - q) + kd * (dq_des - dq)
        # is re-evaluated before every step, as a real motor driver's PD loop
        # would be (PerStepPdBridge only stores the command; one DDS
        # deserialisation per message, as in the stock bridge).
        latest = bridge.latest

        def apply_pd():
            cmd = latest["cmd"]
            if cmd is None:
                return
            q_des, dq_des, kp, kd, tau = cmd
            q = data.sensordata[:nu]
            dq = data.sensordata[nu:2 * nu]
            data.ctrl[:] = tau + kp * (q_des - q) + kd * (dq_des - dq)

        print(f"G1 standing (pelvis pinned, no band); DDS domain {args.domain_id}, interface {args.interface}.")
        # Real-time schedule: step until simulated time catches up with the
        # wall clock, then sleep. (Sleeping once per 5 ms step, as the stock
        # loop does, loses 1-2 ms per step to timer slack on the VM.)
        wall_start, sim_start = time.perf_counter(), data.time
        wall0, sim0, last_report = wall_start, data.time, wall_start
        while running():
            now = time.perf_counter()
            behind = (now - wall_start) - (data.time - sim_start)
            if behind > 0.25:
                # Too far behind to catch up (machine overloaded): drop the backlog.
                wall_start, sim_start = now, data.time
                behind = 0.0
            if behind > 0:
                with lock:
                    apply_pd()
                    mujoco.mj_step(model, data)
            else:
                time.sleep(min(-behind, 0.002))
            if now - last_report > 10.0:
                rate = (data.time - sim0) / (now - wall0)
                print(f"sim speed {rate:.2f}x real time" + ("  (slower than real time: close other programs, "
                      "lower --viewer-fps)" if rate < 0.9 else ""), flush=True)
                wall0, sim0, last_report = now, data.time, now

    def render():
        while viewer.is_running():
            with lock:
                viewer.sync()
            time.sleep(1.0 / args.viewer_fps)

    threads = [threading.Thread(target=simulate)] + ([] if viewer is None else [threading.Thread(target=render)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
