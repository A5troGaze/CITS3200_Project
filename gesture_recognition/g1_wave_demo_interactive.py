"""
g1_wave_demo_interactive.py

Same control loop as g1_wave_demo.py, but using MuJoCo's interactive GLFW
viewer window instead of offscreen rendering to a video file. Use this if
the interactive viewer works on your machine and you want to watch the
robot move live and interact with it (drag it, pause physics, etc).

Usage (macOS requires mjpython, not python):
    mjpython g1_wave_demo_interactive.py /path/to/unitree_g1/scene.xml
"""
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer


def find_actuator(model, keywords):
    for kw in keywords:
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and kw in name.lower():
                return i
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: mjpython g1_wave_demo_interactive.py /path/to/scene.xml")
        sys.exit(1)

    model_path = sys.argv[1]
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    actuator_id = find_actuator(model, ["shoulder", "elbow", "arm"])
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    print(f"Model loaded: {model.nu} actuators, {model.nq} DOFs")
    print(f"Driving actuator #{actuator_id} ('{actuator_name}') with a sine wave")

    center = data.ctrl[actuator_id]
    amplitude = 0.4

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            data.ctrl[actuator_id] = center + amplitude * np.sin(2 * np.pi * 0.3 * t)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
