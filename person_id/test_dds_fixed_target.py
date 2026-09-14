"""MuJoCo-only integration test for the DDS controller."""

import argparse
import time

from mujoco_pose_controller import LEFT_ELBOW, LEFT_SHOULDER_PITCH, MujocoPoseController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    args = parser.parse_args()
    controller = MujocoPoseController(args.domain_id, args.interface)
    try:
        controller.init()
        controller.start()
        print("Connected; applying a conservative left-arm target...")
        controller.set_targets({LEFT_SHOULDER_PITCH: -0.5, LEFT_ELBOW: 0.3})
        time.sleep(3.0)
    except TimeoutError as exc:
        raise SystemExit(f"Could not connect to MuJoCo: {exc}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if controller.ready.is_set():
            controller.return_to_start()
        controller.stop()
    print("Done; confirm the arm moved smoothly and returned.")


if __name__ == "__main__":
    main()
