# Humanoid Project

## Dependencies

### Project Dependencies
- Python >= 3.8
- cyclonedds == 0.10.2
- unitree_sdk2_py 1.0.1
- MediaPipe 1.0.0
- Mujoco 3.11.0
- pinocchio 4.1.0
- GMR 0.2.0


## Development Setup:
### Structure:
```
~/CITS3200/
├── Dependencies/              ← third-party code + Python environment
│   ├── g1-env/                ← Python virtual environment
│   ├── unitree_sdk2_python/   ← cloned from Unitree's GitHub
│   └── GMR/                   ← cloned from YanjieZe/GMR
└── Project/                   ← the actual Git repo (this is what you clone from GitHub)
    ├── Documentation/
    └── Project Files/
    
~/cyclonedds/                  ← kept separate — system-level infrastructure
└── install/
```

### Step 1: Install Git and the Github CLI
---
1. **Install Git**
```bash
sudo apt update             # Refresh Ubuntu's package list
sudo apt install git -y     # Installs Git version control
git --version               # Confirms install
```
---
2. **Set Git identity:**
```bash
git config --global user.name "your_name"                   # Sets your user name
git config --global user.email "your_Github_email@address"  # Sets email address (use the same as your github account)
```
---
3. **Install Github CLI(`gh`)**

Check for the `curl` download tool and installs it if not found:
```bash
type -p curl > /dev/null || sudo apt install curl -y
```

Download's Github's official signing keys and saves them to your system:
```bash
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
```

Makes keys readable by system:
```bash
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
```

Tells Ubuntu package manager about a new source to check for packages. (Github's server, verified using the keys from previous step):
```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
```

Install Github and verify install:
```bash
sudo apt update             # Refresh package list again
sudo apt install gh -y      # Install Github CLI
gh --version                # Confirm installation
```
---
4. **Authenticate with Github:**
```bash
gh auth login
```
This starts an interactive set of prompts. Answer as follows:
| Prompt | Answer |
|---|---|
| What account do you want to log into? | `GitHub.com` |
| What is your preferred protocol for Git operations? | `HTTPS` |
| Authenticate Git with your GitHub credentials? | `Yes` — this is the important one; it tells `gh` to also configure plain `git` commands to use this same login, not just `gh` itself |
| How would you like to authenticate? | `Login with a web browser` |
 
After the last prompt, it will display a short one-time code and open your browser automatically. Paste the code into GitHub's site, log in, and click authorize. From this point on, your credentials are stored securely and reused automatically for every Git operation on this machine — including private repos you have access to.

Afterwards, check you are logged in using:
```bash
gh auth status
```
---
### Step 2: Install Python
1. **Install, Python, pip, venv:**
```bash
sudo apt install python3 python3-pip python3-venv -y        # Install
python3 --version                                           # Verify python3 install
pip3 --version                                              # Verify pip3 install
```
---
### Step 3: Install / Build CycloneDDS (unitree_sdk2_python requirement)
CycloneDDS is a C library — it lives outside any project folder, in your home directory, since it's system-level infrastructure, not a Python package.
1. **Install build tools:**
```bash
sudo apt install cmake build-essential -y
```
---
2. **Clone CycloneDDS and check out exact version according to `unitree_sdk2_python` requirements:**
```bash
cd ~                                                                            # Change directory to root directory
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x   # Clone from Github
cd cyclonedds                                                                   # Change directory to cyclonedds directory
git checkout 0.10.2                                                             # Checkout the correct version
```
---
3. **Build and install it:**
```bash
mkdir build install                                                             # Make build and install directories
cd build                                                                        # Change directory to build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install                                                # Compiles CycloneDDS. Takes a couple minutes
```
Verify contents of install directory:
Should see `bin`, `include`, `lib`, `share`
```bash
ls ~/cyclonedds/install
```
---
4. **Make CYCLONEDDS_HOME permanent:**
```bash
echo 'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"' >> ~/.bashrc
source ~/.bashrc
```
Verify: should print `/home/<your-username>/cyclonedds/install`
```bash
echo $CYCLONEDDS_HOME
```
---
### Step 4: Set Up Project Structure
1. **Create master folder and clone the repo:**
```bash
mkdir ~/CITS3200                                                        # Make master folder
cd ~/CITS3200                                                           # Change directory to master folder (/CITS3200)
git clone https://github.com/A5troGaze/CITS3200_Project.git Project     # Clone online repository, name Repository folder 'Project'
```
---
2. **Create the Dependencies folder:**

This directory will hold all the third party code and the python environment (venv) - Kept separate from the Git repo on purpose so none of it accidentally gets committed.
```bash
mkdir ~/CITS3200/Dependencies
```
---
### Step 5: Clone and Install Project Dependencies
1. **Clone the unitree_sdk2_python repo:**
```bash
cd ~/CITS3200/Dependencies                                              # Change to the Dependency folder
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git    # Clone from Github
```
---
2. **Clone GMR:**
```bash
cd ~/CITS3200/Dependencies                                              # Change to the Dependency folder
git clone https://github.com/YanjieZe/GMR.git                           # Clone from Github
```
---
3. **Create and activate the virtual environment**
```bash
python3 -m venv g1-env                                                  # Create virtual environment called g1-env
source ~/CITS3200/Dependencies/g1-env/bin/activate                      # Activate the virtual environment
```
Prompts should now show `(g1-env)` infront of it

---
4. **Install Unitree SDK2 Python:**
```bash
cd ~/CITS3200/Dependencies/unitree_sdk2_python                          # Change directory to unitree_sdk2_python's clone
pip3 install -e .                                                       # Install
python3 -c "import unitree_sdk2py; print('Unitree SDK OK')"             # Verify install
```
---

5. **Install MediaPipe:**
```bash
pip3 install mediapipe                                                  # Install
python3 -c "import mediapipe; print('MediaPipe OK')"                    # Verify install
```
---

6. **Install Mujoco:**
```bash
pip3 install mujoco                                                     # Install
python3 -c "import mujoco; print('MuJoCo OK')"                          # Verify install
```
---

7. **Install Pinocchio:**
```bash
pip3 install pin                                                        # Install
python3 -c "import pinocchio; print('Pinocchio OK')"                    # Verify install
```
Note: pip package name is `pin`, but you import it as `import pinocchio`.

---
8. **Install GMR:**
```bash
cd ~/CITS3200/Dependencies/GMR                                          # Change to GMR directory inside Dependency directory
pip3 install -e .                                                       # Install, NOTE: Takes longer than others
python3 -c "import general_motion_retargeting; print('GMR OK')"         # Verify install
```
You may see a line saying `xrobotoolkit_sdk not found, skip for now` — that's just an informational notice about an optional VR-streaming feature we're not using. Not an error.

---

### Step 6: Verify All Installs
With `g1-env` active, run:
```bash
python3 -c "
import unitree_sdk2py
import mediapipe
import mujoco
import pinocchio
import general_motion_retargeting
print('All 5 dependencies OK')"
```
If this prints `All 5 dependencies OK` with no errors, the environment is full set up.

## Workflow and Warnings:
- **Every new terminal session**, before running any project Python code:
```bash
  source ~/CITS3200/Dependencies/g1-env/bin/activate
```
- **Never commit anything from `~/CITS3200/Dependencies/`** to Git — it's intentionally outside the `Project` repo folder, so this shouldn't happen by accident, but don't manually copy those files into `Project` either.
- The repo's `.gitignore` already excludes Python cache files, editor settings, OS junk files, and common data/output file types (`.pkl`, `.mp4`, etc.).
- Development work happens on feature branches (e.g. `Gesture-Recog`), not directly on `main`

## Person ID / MuJoCo bridge

This is the extra setup needed for `person_id/leader_pose.py --mujoco` and
`person_id/mujoco_link.py`: mirroring the leader's arms on a simulated G1.
See `person_id/RUNNING_leader_pose.md` for how to actually run it once this
is installed.

### 1. Clone and build unitree_mujoco

```bash
cd ~/CITS3200/Dependencies
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

Use the **Python simulator** (`simulate_python/`), not the C++ one
(`simulate/`): it only needs `unitree_sdk2_python` and the `mujoco` pip
package, both already in this venv from Step 5 above, so there is no extra
C++ toolchain (`unitree_sdk2` C++, libyaml-cpp, a separate MuJoCo binary
download) to install. No build step — it's run directly with `python3`.

### 2. Point its config at the G1

Edit `~/CITS3200/Dependencies/unitree_mujoco/simulate_python/config.py`:

```python
ROBOT = "g1"
# ROBOT_SCENE is derived from ROBOT automatically: "../unitree_robots/g1/scene.xml"
DOMAIN_ID = 1     # matches mujoco_link.py's --dds-domain default
INTERFACE = "lo"  # matches mujoco_link.py's --dds-interface default
ENABLE_ELASTIC_BAND = True  # arm-only mirroring: legs/waist are held, not balanced
```

`unitree_robots/g1/scene.xml` (confirmed by reading the file, not assumed)
includes `g1_29dof.xml` — the same 29-DOF-no-hands variant
`person_id/g1_joint_limits.py` is built from, *not* `scene_23dof.xml`.
Before trusting `leader_pose.py --mujoco` output against this scene, run
`check_dof_count()` (from `g1_joint_limits.py`) against that scene's joint
names (or against a live `LowState_`'s `motor_state` indices) to confirm
they still match — this is exactly the silent-mismatch case that function
exists to catch if the scene file is ever swapped.

### 3. Run it

```bash
cd ~/CITS3200/Dependencies/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
```

A MuJoCo window opens with the G1 standing. Since this pipeline only
commands the arms and holds legs/waist at whatever they currently are
(never balances or walks), enable the elastic band (`ENABLE_ELASTIC_BAND`
above) so the robot hangs instead of falling over: once loaded, press `9`
to activate/release the band, `7` to lower the robot, `8` to lift it
(these bindings are unitree_mujoco's own, not this project's).