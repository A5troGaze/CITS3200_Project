"""
Receives live pose landmarks over UDP and forwards them to the MuJoCo link.

Use this when the webcam runs on Windows but MuJoCo runs in WSL.
"""

import argparse
import json
import socket

from mujoco_link import MujocoLink


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--dds-domain", type=int, default=1, help="CycloneDDS domain ID for the MuJoCo bridge")
    parser.add_argument("--dds-interface", default="lo", help="Network interface for the MuJoCo bridge")
    args = parser.parse_args()

    mujoco_link = MujocoLink(domain_id=args.dds_domain, interface=args.dds_interface)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print(f"Listening for pose landmarks on UDP {args.host}:{args.port}")
    print("Keep Unitree MuJoCo running in the other WSL terminal.")

    try:
        while True:
            data, addr = sock.recvfrom(65535)
            packet = json.loads(data.decode("utf-8"))

            frame_id = packet.get("frame_id", -1)
            timestamp_ms = packet.get("timestamp_ms", 0)
            landmarks = packet.get("landmarks_world_m", [])

            mujoco_link.update_pose(frame_id, timestamp_ms, landmarks)

    except KeyboardInterrupt:
        print("Stopped live pose receiver.")

    finally:
        sock.close()
        mujoco_link.close()


if __name__ == "__main__":
    main()
