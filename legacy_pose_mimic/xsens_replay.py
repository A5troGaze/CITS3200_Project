"""
Optional input: drive the G1 from an Xsens mocap suit recording (BVH)
through GMR's own Xsens path, into the same controller as the camera
pipeline.

GMR already supports Xsens BVH exports (utils/xsens.py, IK config
ik_configs/bvh_xsens_to_g1.json), with full joint orientations from the
suit, so none of the MediaPipe adapter is needed here: GMR's loader ->
GMR per frame -> joint name -> motor index, clamped to the real limits
(gmr_retarget.py) -> MujocoPoseController / ArmSdkPublisher.

    # Offline check on GMR's sample recording:
    python3 xsens_replay.py ~/CITS3200/Dependencies/GMR/assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh --dry-run

    # Into the running sim (elastic band on):
    python3 xsens_replay.py recording.bvh

Only 3ds Max-format BVH exports (GMR's "3DSM" format, cm units -> --scale
0.01) are supported, because that is what GMR's loader supports. The root
pose is not used and the legs are not commanded unless --legs is given
(sim with the band only), as with the camera pipeline.
"""

import argparse
import contextlib
import io
import time

import numpy as np

from gmr_retarget import MOTOR_JOINT_NAMES, clamp_to_limits, commanded_joint_names
from g1_joint_limits import G1_29DOF_JOINT_LIMITS


def load_bvh(path, scale=0.01, start=None, end=None):
    """Xsens BVH -> GMR frames. Returns (frames, human_height_m, frame_time_s).

    Same steps as GMR's utils/xsens.load_xsens_file (BVHParser, zxy axis
    order, forward kinematics, the LeftFootMod/RightFootMod keys its
    bvh_xsens_to_g1 config expects), minus its OffsetManager: that class
    lives in a PyQt6 curve-editor GUI module, and PyQt6 is not a declared
    GMR dependency. Without an offsets.json it returns all-zero offsets,
    which is what is used here."""
    import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils
    from general_motion_retargeting.utils.xsens_vendor.BVHParser import Anim, BVHParser

    parser = BVHParser(axis_order="zxy", scale=scale)
    with open(path, "r") as f:
        text = f.read()
    with contextlib.redirect_stdout(io.StringIO()):
        rotations, _ = parser.parse(text, start=start, end=end, reset_to_zero=False)
        quats, positions, offsets, parents = parser._MOTION_data_post_processing(
            rotations, np.copy(parser.positions), reset_to_zero=True)
    anim = Anim(quats, positions, offsets, parents, parser.names)
    global_rot, global_pos = lafan_utils.quat_fk(anim.quats, anim.pos, anim.parents)
    frames = []
    for k in range(anim.pos.shape[0]):
        frame = {bone: (global_pos[k, i], global_rot[k, i]) for i, bone in enumerate(anim.bones)}
        frame["LeftFootMod"] = (np.array(frame["LeftAnkle"][0]), frame["LeftAnkle"][1])
        frame["RightFootMod"] = (np.array(frame["RightAnkle"][0]), frame["RightAnkle"][1])
        frames.append(frame)
    last = frames[-1]
    height = last["Head_end_site"][0][2] - min(last["LeftToe_end_site"][0][2], last["RightToe_end_site"][0][2])
    return frames, float(height), parser.frame_time


class XsensRetargeter:
    """GMR with its bvh_xsens -> unitree_g1 config; per-frame solve mapped to
    motor indices by joint name."""

    def __init__(self, human_height, waist="off", legs=False):
        with contextlib.redirect_stdout(io.StringIO()):
            from general_motion_retargeting import GeneralMotionRetargeting

            self.gmr = GeneralMotionRetargeting(src_human="bvh_xsens", tgt_robot="unitree_g1",
                                                actual_human_height=human_height, verbose=False)
        model = self.gmr.model
        self.qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in MOTOR_JOINT_NAMES}
        self.commanded = {G1_29DOF_JOINT_LIMITS[n].index: n for n in commanded_joint_names(waist, legs)}
        self.waist = waist

    def step(self, frame):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            qpos = self.gmr.retarget(frame)
        ms = (time.perf_counter() - t0) * 1000.0
        if not np.all(np.isfinite(qpos)):
            raise ValueError("GMR returned a non-finite solution")
        q = {name: clamp_to_limits(name, qpos[adr]) for name, adr in self.qpos_adr.items()}
        if self.waist == "yaw":
            q["waist_roll_joint"] = q["waist_pitch_joint"] = 0.0
        targets = {idx: q[name] for idx, name in self.commanded.items()}
        return targets, q, ms


def main():
    from person_id_replay import add_pipeline_args, build_publisher, shutdown_publisher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bvh_file")
    parser.add_argument("--scale", type=float, default=0.01, help="BVH length units to metres (cm: 0.01)")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="No DDS: print targets only")
    add_pipeline_args(parser)
    args = parser.parse_args()
    if args.mirror:
        raise SystemExit("--mirror is not supported for mocap replay")

    frames, height, frame_time = load_bvh(args.bvh_file, args.scale, args.start, args.end)
    print(f"Loaded {len(frames)} frames at {1 / frame_time:.0f} Hz, performer height {height:.2f} m.")
    retargeter = XsensRetargeter(height, waist=args.waist, legs=args.legs)

    pub = None
    lat = []
    try:
        pub = build_publisher(args)
        start = time.monotonic()
        k = 0
        while k < len(frames):
            targets, _, ms = retargeter.step(frames[k])
            lat.append(ms)
            if pub is not None:
                pub.set_targets(targets)
                # Real time: skip ahead to the frame for "now" (mocap is 60-120 Hz).
                k = max(k + 1, int((time.monotonic() - start) / frame_time))
            else:
                if k % 60 == 0:
                    print(f"frame {k}: gmr {ms:5.1f} ms " +
                          " ".join(f"{i}:{v:+.2f}" for i, v in sorted(targets.items())))
                k += 1
        print(f"Done: GMR per frame median {np.median(lat):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms "
              f"over {len(lat)} solved frames.")
    except KeyboardInterrupt:
        print("Interrupted.")
    except TimeoutError as exc:
        raise SystemExit(f"Could not connect: {exc}")
    finally:
        shutdown_publisher(pub)


if __name__ == "__main__":
    main()
