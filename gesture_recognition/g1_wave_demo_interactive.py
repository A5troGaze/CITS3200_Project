"""
g1_wave_demo_interactive.py

Same control loop as g1_wave_demo.py, but using MuJoCo's interactive GLFW
viewer window instead of offscreen rendering to a video file. Use this if
the interactive viewer works on your machine and you want to watch the
robot move live and interact with it (drag it, pause physics, etc).

Usage (macOS requires mjpython, not python):
    mjpython g1_wave_demo_interactive.py /path/to/unitree_g1/scene.xml
"""
#== Import Dependencies ===============================================================
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer


def find_actuator(model, keywords):
    for kw in keywords:                                                         # For each decribed keyword
        for i in range(model.nu):                                               # For each actuator that the model (G1) possesses
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and kw in name.lower():                                     # If the actuator name and the keyword match
                return i                                                        # Return the index of the actuator
    return 0                                                                    # Else return 0 if matching actuator cannot be found

#== Define app, behaviour ==============================================================
def main():
    #== Error Catch: Program needs a model to run ======================================
    if len(sys.argv) < 2:
        print("Usage: python g1_wave_demo_interactive.py /path/to/scene.xml")
        sys.exit(1)

    #== Define Model from model path ===================================================
    model_path = sys.argv[1]                                                    # Get model path for argument
    model = mujoco.MjModel.from_xml_path(model_path)                            # Extract model blueprint
    data = mujoco.MjData(model)                                                 # Build live state from blueprint

    #== Movement definition ============================================================
    actuator_id = find_actuator(model, ["shoulder", "elbow", "arm"])            # Find actuator index
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) # Find actuator name
    print(f"Model loaded: {model.nu} actuators, {model.nq} DOFs")               # Output number of actuators and degrees of freedom
    print(f"Driving actuator #{actuator_id} ('{actuator_name}') with a sine wave")

    center = data.ctrl[actuator_id]                                             # Control actuator matchin ID
    amplitude = 0.4                                                             # Define radians of movement

    #== Simulation Loop ===============================================================
    with mujoco.viewer.launch_passive(model, data) as viewer:                   # Launch window
        start = time.time()
        while viewer.is_running():                                              # While window is not closed
            t = time.time() - start                                             # Compute elapsed time
            data.ctrl[actuator_id] = center + amplitude * np.sin(2 * np.pi * 0.3 * t)   # write new target angle for actuator
            mujoco.mj_step(model, data)                                         # Advance engine by one time step
            viewer.sync()                                                       # Push to the window

#== Run Program ========================================================================
if __name__ == "__main__":
    main()
