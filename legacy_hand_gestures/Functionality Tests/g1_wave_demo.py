"""
g1_wave_demo.py

Minimal MuJoCo bring-up script for the Unitree G1: loads the model, steps
physics while driving one arm actuator with a sine wave, and renders the
result to an MP4 file using MuJoCo's offscreen renderer. This proves the
same control loop (data.ctrl -> mj_step -> render) that gesture recognition
will plug into later, without depending on the interactive GLFW viewer
window (which has a known compatibility issue on some Mac setups).

Usage:
    python g1_wave_demo.py /path/to/unitree_g1/scene.xml

Output:
    g1_wave_demo.mp4 in the current directory.

If you'd rather use the interactive window and it works on your machine,
see g1_wave_demo_interactive.py instead (same logic, uses mujoco.viewer +
mjpython).
"""
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio


def find_actuator(model, keywords):
    """Return the id of the first actuator whose name contains one of the
    given keywords, falling back to actuator 0 if none match. This keeps
    the script working across the different G1 variants (23dof/29dof)
    without hardcoding exact joint names."""
    for kw in keywords:
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and kw in name.lower():
                return i
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python g1_wave_demo.py /path/to/scene.xml")
        sys.exit(1)

    model_path = sys.argv[1]
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    actuator_id = find_actuator(model, ["shoulder", "elbow", "arm"])
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    print(f"Model loaded: {model.nu} actuators, {model.nq} DOFs")
    print(f"Driving actuator #{actuator_id} ('{actuator_name}') with a sine wave")

    center = data.ctrl[actuator_id]
    amplitude = 0.4  # radians -- small and safe, won't fling the arm around

    duration = 6.0  # seconds of simulated time to record
    fps = 30
    frame_dt = 1.0 / fps

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    next_frame_time = 0.0

    while data.time < duration:
        data.ctrl[actuator_id] = center + amplitude * np.sin(2 * np.pi * 0.3 * data.time)
        mujoco.mj_step(model, data)

        if data.time >= next_frame_time:
            renderer.update_scene(data)
            frames.append(renderer.render())
            next_frame_time += frame_dt

    renderer.close()

    out_path = "g1_wave_demo.mp4"
    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"Wrote {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
