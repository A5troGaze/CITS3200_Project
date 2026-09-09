# Topics to bring up
## What is CycloneDDS
- Cyclonedds is a middleware Data Distribution Service implementation. It works in real time to transport data/commands to and from the source and origin. Think of it as a messenger between our machine and the robot.

## Mujoco
- The version of mujoco we have been using up until now is not the full version that 'speaks with cyclonedds'. We are using the pip installed version not the git cloned version. The git cloned version has the ability to receive commands from cyclonedds the same way the robot would.
- The one we have been using is nothing but a physics engine controlled with python bindings. It does not receive anything from a network interface. That is not what we want.

## Unitree SDK
- The SDK now needs to be implemented in our code to send commands to a real Mujoco instance.
- Where the trouble comes in: Christo is the only one with a linux machine. CycloneDDS etc. can cause trouble with VMs. Which means Christo needs to test everything on his device.

## Merging, Testing
### Project Structure:

Ideal project structure going forward (may need to be tweaked according to how `person_id` team's folder structure looks)
```
~/CITS3200/
├── Dependencies/
│   ├── g1-env/                     # unchanged
│   ├── unitree_sdk2_python/        # unchanged
│   ├── GMR/                        # unchanged
│   ├── Models/                     # unchanged
│   ├── mujoco_menagerie/           # unchanged
│   └── unitree_mujoco/             # NEW — clone; only simulate_python/ gets used
│       ├── simulate_python/
│       │   ├── unitree_mujoco.py   # the DDS-aware sim you actually run
│       │   └── config.py           # ROBOT, DOMAIN_ID, INTERFACE settings
│       └── unitree_robots/         # MJCF models this repo ships for its own topic mapping
└── Project/
    ├── Documentation/
    ├── gesture_library/
    ├── gesture_recognition/
    │    ├── Launch Documentation/
    │    ├── Testing Documentation/
    │    ├── data/
    │    ├── g1_wave_demo.py             # unchanged — keep as the milestone-1 reference; superseded by unitree_mujoco for real DDS-driven control
    │    └── etc.
    ├── person_id/
    └── Kinematics_Reference/

~/cyclonedds/install/               # unchanged, already built (0.10.2) — each teammate needs their own
```
### Gesture Team Merging:
```
Simulation -> Gesture -> main
```

### Pose Team Merging:
```
mujoco-connection-person -> person_id -> main   #???
```

### Install CycloneDDS and unitree_mujoco if not done yet:
1. Build CycloneDDS from source, if not already done on that machine
    ```bash
    cd ~
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
    cd cyclonedds && mkdir build install && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    cmake --build . --target install
    ```

2. Point the environment at that build permanently
    ```bash
    echo 'export CYCLONEDDS_HOME=~/cyclonedds/install' >> ~/.bashrc
    source ~/.bashrc
    ```

3. Install unitree_sdk2_python into the venv, if not already
- This pip links python to the Unitree SDK
    ```bash
    source ~/CITS3200/Dependencies/g1-env/bin/activate
    cd ~/CITS3200/Dependencies/unitree_sdk2_python
    pip install -e .
    ```

4. Clone the DDS-aware simulator
- Missing piece: This is the bridge that listens for the commands from CycloneDDS
    ```bash
    cd ~/CITS3200/Dependencies
    git clone https://github.com/unitreerobotics/unitree_mujoco.git
    ```

5. Point the sim at the G1 and set it to local-only mode
- Edit `~/CITS3200/Dependencies/unitree_mujoco/simulate_python/config.py`:
    ```python
    ROBOT = "g1"          # defaults to "go2" — must change
    DOMAIN_ID = 1          # must match whatever domain your control code publishes on
    INTERFACE = "lo"       # "lo" = loopback, i.e. sim and control code on the same machine
    ```

6. Run:
    ```bash
    cd ~/CITS3200/Dependencies/unitree_mujoco/simulate_python
    python unitree_mujoco.py
    ```

## GMR
- Where are we sitting on GMR? Will we implement it or should we have a discussion with Oliver regarding scope?

## Testing Role going forward
- Christo and Dylan needs to be in testing role as the only one with access to a physical machine that can actually handle and send CycloneDDS commands is Christo.
- For that Christo and Dylan needs everyone else to stay on top of things. Otherwise we are going to be left solo scrambling at the end of semester.
- Ideally the group should be finished by a decent time in advance to give us proper breather and time to look over things and catch any errors before cutoff dates. Hence:
    - **WHOLE** Group needs to be finished by **_7th October_** to allow for an 'oh shit' buffer
    - Development needs to be finished by **_1st October_** to allow enough time for real-life testing
        - Need to discuss with Oliver for lab access times