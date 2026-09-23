"""Manual MuJoCo integration test for the CycloneDDS path (SIMULATOR ONLY).

Start unitree_mujoco first (elastic band on), then:

    python3 test_dds_fixed_target.py

Sends fixed arm and waist targets through MujocoPoseController, reads the
joint positions back from rt/lowstate, and checks they settle within
--tolerance of the targets; then returns to the start pose and checks
that too. Prints the numbers and exits non-zero on failure.
Never run this against a real robot.
"""

import argparse
import sys
import time

import numpy as np

from gmr_retarget import MOTOR_JOINT_NAMES
from mujoco_pose_controller import MujocoPoseController

TARGETS = {
    12: 0.30,    # waist yaw
    15: -1.20,   # left shoulder pitch (arm forward/up)
    16: 0.40,    # left shoulder roll
    18: 0.30,    # left elbow
    22: -0.50,   # right shoulder pitch
    23: -0.80,   # right shoulder roll (arm out)
    25: 1.20,    # right elbow
}


def settle_and_report(ctl, targets, label, tolerance, settle_s, window_s=1.0):
    """Wait settle_s, then average rt/lowstate over the last window_s: on the
    elastic band the G1 hangs free and swings a little whenever an arm
    moves, so a single sample mixes that swing into the tracking error."""
    time.sleep(max(0.0, settle_s - window_s))
    samples = []
    end = time.monotonic() + window_s
    while time.monotonic() < end:
        samples.append(ctl.measured_positions())
        time.sleep(0.01)
    samples = np.array(samples)
    q, spread = samples.mean(axis=0), samples.std(axis=0)
    worst = 0.0
    print(f"\n{label}: joint, target, measured mean (+- std over {window_s:.0f} s), error (rad)")
    for i, target in targets.items():
        err = q[i] - target
        worst = max(worst, abs(err))
        print(f"  {i:2d} {MOTOR_JOINT_NAMES[i]:28s} {target:+.3f} {q[i]:+.3f} (+-{spread[i]:.3f}) {err:+.4f}")
    ok = worst <= tolerance
    print(f"  worst |error| {worst:.4f} rad ({np.degrees(worst):.2f} deg) -> {'PASS' if ok else 'FAIL'}"
          f" (tolerance {tolerance} rad)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--settle", type=float, default=3.0)
    args = parser.parse_args()
    controller = MujocoPoseController(args.domain_id, args.interface)
    ok = True
    try:
        controller.init()
        controller.start()
        start = controller.hold_positions.copy()
        print("Connected; start pose captured from rt/lowstate. Holding 3 s so the robot settles on the band.")
        time.sleep(3.0)
        controller.set_targets(TARGETS)
        ok &= settle_and_report(controller, TARGETS, "Fixed targets", args.tolerance, args.settle)
        controller.clear_targets()
        ok &= settle_and_report(controller, {i: start[i] for i in TARGETS}, "Back to start",
                                args.tolerance, args.settle)
        print(f"\nPublished {controller.published} LowCmd messages.")
    except TimeoutError as exc:
        raise SystemExit(f"Could not connect to MuJoCo: {exc}")
    except KeyboardInterrupt:
        print("Interrupted.")
        ok = False
    finally:
        if controller.ready.is_set():
            controller.return_to_start()
        controller.stop()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
