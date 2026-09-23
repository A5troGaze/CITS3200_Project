"""
Headless MuJoCo helpers on unitree_mujoco's own G1 model.

* load_sim_model(fixed_base=True): the exact MJCF the simulator runs
  (unitree_robots/g1/scene.xml), optionally with the floating base removed
  so the pelvis is pinned at its rest pose (balance is out of scope).
* PdSim: steps that model with the simulator's PD law
  (unitree_sdk2py_bridge.LowCmdHandler: ctrl = kp*(q_des - q) + kd*(0 - dq),
  clipped to the actuator ctrlrange) and its SIMULATE_DT, so the "dynamic"
  mimicry check sees the same tracking error the simulator would.
* segment_vectors: robot upper-arm / forearm directions in the robot torso
  frame and the torso orientation relative to the pelvis, from MuJoCo body
  positions. This is what "the robot copies the person" is measured on.

No DDS, no viewer. mujoco.Renderer is only imported by render helpers.
"""

import os

import numpy as np

from g1_gains import KD, KP

UNITREE_MUJOCO = os.path.expanduser(
    os.environ.get("UNITREE_MUJOCO_DIR", "~/CITS3200/Dependencies/unitree_mujoco"))
SCENE_XML = os.path.join(UNITREE_MUJOCO, "unitree_robots", "g1", "scene.xml")
SIMULATE_DT = 0.005  # unitree_mujoco simulate_python/config.py

SIDES = ("left", "right")


def load_sim_model(fixed_base=True, xml=SCENE_XML):
    import mujoco

    if not fixed_base:
        return mujoco.MjModel.from_xml_path(xml)
    spec = mujoco.MjSpec.from_file(xml)
    pelvis = spec.body("pelvis")
    for joint in list(pelvis.joints):
        spec.delete(joint)
    return spec.compile()


def joint_qpos_adr(model):
    import mujoco

    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            out[model.joint(j).name] = int(model.jnt_qposadr[j])
    return out


def set_joint_positions(model, data, q_named):
    import mujoco

    adr = joint_qpos_adr(model)
    for name, value in q_named.items():
        data.qpos[adr[name]] = value
    mujoco.mj_kinematics(model, data)


def segment_vectors(model, data):
    """Returns (upper, fore, torso_rel) where upper/fore are {side: unit
    vector in the robot torso frame} and torso_rel is the 3x3 torso
    orientation relative to the pelvis. Requires kinematics to be current."""
    r_torso = data.body("torso_link").xmat.reshape(3, 3)
    r_pelvis = data.body("pelvis").xmat.reshape(3, 3)
    upper, fore = {}, {}
    for side in SIDES:
        s = data.body(f"{side}_shoulder_roll_link").xpos
        e = data.body(f"{side}_elbow_link").xpos
        w = data.body(f"{side}_wrist_roll_link").xpos
        upper[side] = r_torso.T @ ((e - s) / np.linalg.norm(e - s))
        fore[side] = r_torso.T @ ((w - e) / np.linalg.norm(w - e))
    return upper, fore, r_pelvis.T @ r_torso


def angle_deg(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class GravityComp:
    """Gravity torque per motor (N*m, motor-index order) for a joint
    configuration of the fixed-base G1, pelvis upright. Sent as the LowCmd
    feed-forward `tau`, so the PD term only has to correct tracking error
    instead of also holding the arms and upper body up against gravity
    (with the SDK's KP = 40, gravity alone sags a horizontal arm or a
    leaning torso by 10-30 degrees)."""

    def __init__(self, model=None):
        import mujoco

        self._mujoco = mujoco
        self.model = model if model is not None else load_sim_model(fixed_base=True)
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)])
        self.dadr = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)])

    def __call__(self, q_motor):
        d = self.data
        d.qpos[self.qadr] = q_motor
        d.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, d)
        return d.qfrc_bias[self.dadr].copy()


class PdSim:
    """Fixed-base G1 stepped with unitree_mujoco's PD law.

    Joints without a target are held at their start position with the same
    gains (what mujoco_pose_controller does in its default hold mode)."""

    def __init__(self, model=None, dt=SIMULATE_DT, kp=KP, kd=KD, gravity_comp=True):
        import mujoco

        self._mujoco = mujoco
        self.model = model if model is not None else load_sim_model(fixed_base=True)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        self.adr = joint_qpos_adr(self.model)
        self.nu = self.model.nu
        # Actuator i drives motor i (checked in tests/test_g1_mapping.py).
        self.act_qadr = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.nu)])
        self.act_dadr = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]] for i in range(self.nu)])
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.lo = self.model.actuator_ctrlrange[:, 0]
        self.hi = self.model.actuator_ctrlrange[:, 1]
        mujoco.mj_forward(self.model, self.data)
        self.q_des = self.data.qpos[self.act_qadr].copy()
        self.gravity = GravityComp(self.model) if gravity_comp else None

    def set_targets(self, targets):
        """{motor_index: q_des}."""
        for idx, q in targets.items():
            self.q_des[idx] = q

    def step(self, n=1):
        m, d = self.model, self.data
        for _ in range(n):
            q = d.qpos[self.act_qadr]
            dq = d.qvel[self.act_dadr]
            tau = self.gravity(q) if self.gravity is not None else 0.0
            d.ctrl[:] = np.clip(tau + self.kp * (self.q_des - q) - self.kd * dq, self.lo, self.hi)
            self._mujoco.mj_step(m, d)

    def joint_positions(self):
        return {name: float(self.data.qpos[a]) for name, a in self.adr.items()}
