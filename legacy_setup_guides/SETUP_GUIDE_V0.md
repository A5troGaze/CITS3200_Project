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
- MediaPipe Hand Landmarker model (`hand_landmarker.task`)


## Development Setup:
### Structure:
```
~/CITS3200/
├── Dependencies/              ← third-party code + Python environment
│   ├── g1-env/                ← Python virtual environment
│   ├── unitree_sdk2_python/   ← cloned from Unitree's GitHub
│   ├── GMR/                   ← cloned from YanjieZe/GMR
│   ├── mujoco_menagerie       ← cloned from google-deepmind/mujoco_menagerie
│   └── Models/                ← downloaded model files (e.g. MediaPipe hand landmarker)
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
3. **Clone Mujoco Menagerie
```bash
cd ~/CITS3200/Dependencies                                              # Change to the Dependency folder
git clone https://github.com/google-deepmind/mujoco_menagerie.git       # Clone from Github
```
---
4. **Create and activate the virtual environment**
```bash
python3 -m venv g1-env                                                  # Create virtual environment called g1-env
source ~/CITS3200/Dependencies/g1-env/bin/activate                      # Activate the virtual environment
```
Prompts should now show `(g1-env)` infront of it

---
5. **Install Unitree SDK2 Python:**
```bash
cd ~/CITS3200/Dependencies/unitree_sdk2_python                          # Change directory to unitree_sdk2_python's clone
pip3 install -e .                                                       # Install
python3 -c "import unitree_sdk2py; print('Unitree SDK OK')"             # Verify install
```
---

6. **Install MediaPipe:**
```bash
pip3 install mediapipe                                                  # Install
python3 -c "import mediapipe; print('MediaPipe OK')"                    # Verify install
```
---

7. **Download the MediaPipe Hand Landmarker model:**

MediaPipe's hand-tracking package (installed above) only provides the *code* — the pretrained model file itself is a separate binary that must be downloaded. It's large and externally sourced, so like the rest of `Dependencies/`, it's kept outside the Git repo rather than committed.

```bash
mkdir -p ~/CITS3200/Dependencies/Models                                                                                                                          # Create a folder to hold downloaded model files
curl -o ~/CITS3200/Dependencies/Models/hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task   # Download the model
ls -la ~/CITS3200/Dependencies/Models/hand_landmarker.task                                                                                                       # Verify: should be several MB, not 0 bytes
```
Any code using MediaPipe's hand landmark detection loads this file via its path (`~/CITS3200/Dependencies/Models/hand_landmarker.task`) — it doesn't come from `pip3 install mediapipe` alone.

---

8. **Install Mujoco:**
```bash
pip3 install mujoco                                                     # Install
python3 -c "import mujoco; print('MuJoCo OK')"                          # Verify install
```
---

9. **Install Pinocchio:**
```bash
pip3 install pin                                                        # Install
python3 -c "import pinocchio; print('Pinocchio OK')"                    # Verify install
```
Note: pip package name is `pin`, but you import it as `import pinocchio`.

---
10. **Install GMR:**
```bash
cd ~/CITS3200/Dependencies/GMR                                          # Change to GMR directory inside Dependency directory
pip3 install -e .                                                       # Install, NOTE: Takes longer than others
python3 -c "import general_motion_retargeting; print('GMR OK')"         # Verify install
```
You may see a line saying `xrobotoolkit_sdk not found, skip for now` — that's just an informational notice about an optional VR-streaming feature we're not using. Not an error.

---

### Step 6: Verify All Installs
1. With `g1-env` active, run:
```bash
python3 -c "
import unitree_sdk2py
import mediapipe
import mujoco
import pinocchio
import general_motion_retargeting
print('All 5 dependencies OK')"
```
If this prints `All 5 dependencies OK` with no errors, the Python packages are fully set up.

2. Confirm the hand landmark model file is present (this isn't a Python import, so it isn't covered by the check above):
```bash
ls -la ~/CITS3200/Dependencies/Models/hand_landmarker.task
```

3. Confirm the mujoco menagerie model and scene file exists (also not covered by check above)
```bash
ls -la ~/CITS3200/Dependencies/mujoco_menagerie/unitree_g1/g1_with_hands.xml        # Check model file
ls -la ~/CITS3200/Dependencies/mujoco_menagerie/unitree_g1/scene_with_hands.xml     # Check scene file
```

If both checks pass, the environment is fully set up.

## Workflow and Warnings:
### 1. **Every new terminal session**, before running any project Python code:
```bash
source ~/CITS3200/Dependencies/g1-env/bin/activate
```
### 2.**Never commit anything from `~/CITS3200/Dependencies/`** to Git
- It is intentionally outside the `Project` repo folder, so this shouldn't happen by accident, but don't manually copy those files into `Project` either. This includes `Dependencies/Models/` — model files are downloaded once per machine, not tracked in Git.
- The repo's `.gitignore` already excludes Python cache files, editor settings, OS junk files, and common data/output file types (`.pkl`, `.mp4`, etc.).
### 3. Development work happens on feature branches (e.g. `Gesture-Recog`), not directly on `main`
To switch and create a new branch:
```bash
git checkout branch-to-branch-from        # Switch to the branch you wish to branch off from
git checkout -b new-branch-name           # Create new branch and switch to it
```
