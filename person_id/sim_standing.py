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
    parser.add_argument("--band", action="store_true",
                        help="free-floating G1 on unitree_mujoco's elastic band instead of a pinned pelvis "
                             "(for --balance rl_lab): 9 releases the band gradually / re-attaches it, 7/8 lift/lower")
    parser.add_argument("--rl-lab", action="store_true",
                        help="implies --band: run unitree_rl_lab's policy inside the sim to balance the "
                             "robot (like Unitree's on-board locomotion) and accept arm commands on "
                             "rt/arm_sdk. The band fades out at 6 s unless --release-band-after is given")
    parser.add_argument("--release-band-after", type=float, default=None,
                        help="with --band: release the band automatically after this many seconds")
    parser.add_argument("--headless", action="store_true",
                        help="no viewer window (e.g. with --record-demo, which renders its own view)")
    args = parser.parse_args()
    if args.rl_lab:
        args.band = True
        if args.release_band_after is None:
            args.release_band_after = 6.0
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

    from unitree_sdk2py_bridge import ElasticBand

    model = load_sim_model(fixed_base=not args.band)
    band = ElasticBand() if args.band else None
    band_body = model.body("torso_link").id
    release = {"t0": None}
    if band is not None:
        # Stock band length 0 leaves the G1 hanging ~0.4 m above the floor, so
        # releasing it drops the robot onto the balance policy. Start with the
        # length at which the band carries ~80 % of the weight with the robot
        # in the rl_lab policy's standing pose (knees bent), feet on the floor.
        from rl_lab_policy import DEFAULT_POSE_MOTOR

        probe = mujoco.MjData(model)
        probe.qpos[:] = model.qpos0
        for i in range(model.nu):
            probe.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] = DEFAULT_POSE_MOTOR[i]
        mujoco.mj_forward(model, probe)
        foot_z = min(probe.body(b).xpos[2] for b in ("left_ankle_roll_link", "right_ankle_roll_link"))
        torso_z = probe.xpos[band_body][2] - (foot_z - 0.045)
        weight = float(model.body_subtreemass[1]) * 9.81
        stiffness0 = band.stiffness
        band.length = max(0.0, (band.point[2] - torso_z) - 0.8 * weight / stiffness0)

        # Releasing fades the band out over 2 s instead of cutting it, so the
        # balance policy takes the weight gradually. 9 starts a release (or
        # re-attaches the band), 7/8 lift/lower as in unitree_mujoco.
        def key_callback(key):
            glfw = mujoco.glfw.glfw
            if key == glfw.KEY_9:
                if band.enable and release["t0"] is None:
                    release["t0"] = data.time
                    print(f"Releasing the band (2 s) at t={data.time:.1f} s", flush=True)
                else:
                    band.enable, band.stiffness, release["t0"] = True, stiffness0, None
            else:
                band.MujuocoKeyCallback(key)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nu = model.nu
    lock = threading.Lock()
    if args.headless:
        viewer = None
    elif band is not None:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    else:
        viewer = mujoco.viewer.launch_passive(model, data)
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
        onboard = None
        if args.rl_lab:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

            from sim_onboard import OnboardBalance

            onboard = OnboardBalance(model, data)
            arm_sub = ChannelSubscriber("rt/arm_sdk", LowCmd_)
            arm_sub.Init(onboard.on_arm_sdk, 10)

        def apply_pd():
            if onboard is not None:
                # The on-board policy owns rt/lowcmd's job; arms come in on rt/arm_sdk.
                onboard.control()
                return
            cmd = latest["cmd"]
            if cmd is None:
                return
            q_des, dq_des, kp, kd, tau = cmd
            q = data.sensordata[:nu]
            dq = data.sensordata[nu:2 * nu]
            data.ctrl[:] = tau + kp * (q_des - q) + kd * (dq_des - dq)

        sim_t0 = data.time
        fell = False
        what = ("balanced by the rl_lab policy (on-board), arms on rt/arm_sdk; band fades out at "
                f"{args.release_band_after:.0f} s" if onboard is not None else
                "on the elastic band (9 releases it)" if band is not None else "standing (pelvis pinned, no band)")
        print(f"G1 {what}; DDS domain {args.domain_id}, interface {args.interface}.", flush=True)
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
                    if band is not None:
                        if args.release_band_after is not None and band.enable and release["t0"] is None \
                                and data.time - sim_t0 > args.release_band_after:
                            release["t0"] = data.time
                            print(f"Releasing the band (2 s) at t={data.time:.1f} s", flush=True)
                        if release["t0"] is not None and band.enable:
                            fade = 1.0 - (data.time - release["t0"]) / 2.0
                            band.stiffness = stiffness0 * max(fade, 0.0)
                            if fade <= 0.0:
                                band.enable = False
                                print(f"Band released at t={data.time:.1f} s", flush=True)
                        force = band.Advance(data.qpos[:3], data.qvel[:3]) if band.enable else np.zeros(3)
                        # The band only ever pulls up (the stock formula can push down).
                        data.xfrc_applied[band_body, :3] = force if force[2] > 0 else 0.0
                    mujoco.mj_step(model, data)
                    if band is not None and not fell and data.qpos[2] < 0.5:
                        fell = True
                        print(f"ROBOT FELL at sim t={data.time:.1f} s", flush=True)
            else:
                time.sleep(min(-behind, 0.002))
            if now - last_report > 10.0:
                rate = (data.time - sim0) / (now - wall0)
                extra = ""
                if band is not None:
                    z = data.qpos[2]
                    extra = f"  pelvis height {z:.2f} m" + ("  (ROBOT HAS FALLEN)" if z < 0.5 else "")
                print(f"sim speed {rate:.2f}x real time" + ("  (slower than real time: close other programs, "
                      "lower --viewer-fps)" if rate < 0.9 else "") + extra, flush=True)
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
