"""unitree_rl_lab balance + person_id arms: policy wiring checked against
rl_lab's own deploy.yaml, the controller's phases (as in the gesture
team's rl_lab_walking_test.py), and an offline balance check."""

import os

import numpy as np
import pytest

import rl_lab_policy as rp

pytestmark = pytest.mark.skipif(not os.path.exists(rp.POLICY_PATH), reason="unitree_rl_lab not cloned")


def test_constants_match_deploy_yaml():
    yaml = pytest.importorskip("yaml")
    with open(rp.DEPLOY_YAML) as f:
        cfg = yaml.safe_load(f)
    assert list(rp.JOINT_IDS_MAP) == cfg["joint_ids_map"]
    assert np.allclose(rp.STIFFNESS, cfg["stiffness"]) and np.allclose(rp.DAMPING, cfg["damping"])
    assert np.allclose(rp.DEFAULT_JOINT_POS, cfg["default_joint_pos"])
    act = cfg["actions"]["JointPositionAction"]
    assert np.allclose(act["scale"], rp.ACTION_SCALE) and np.allclose(act["offset"], cfg["default_joint_pos"])
    assert cfg["step_dt"] == rp.STEP_DT
    obs = cfg["observations"]
    assert list(obs) == ["base_ang_vel", "projected_gravity", "velocity_commands",
                         "joint_pos_rel", "joint_vel_rel", "last_action"]
    assert all(t["history_length"] == rp.HISTORY_LENGTH for t in obs.values())
    assert np.allclose(obs["base_ang_vel"]["scale"], rp.ANG_VEL_SCALE)
    assert np.allclose(obs["joint_vel_rel"]["scale"], rp.JOINT_VEL_SCALE)


def test_policy_io_and_arm_override():
    pol = rp.RlLabPolicy()
    q = rp.DEFAULT_POSE_MOTOR.copy()
    out = pol.step(q, np.zeros(29), [1, 0, 0, 0], np.zeros(3))
    assert out.shape == (29,) and np.all(np.isfinite(out))
    # At its default pose, upright and still, the policy stays near the default pose.
    assert np.max(np.abs(out - rp.DEFAULT_POSE_MOTOR)) < 0.5
    override = {i: 0.4 for i in rp.ARM_MOTORS}
    out = pol.step(q, np.zeros(29), [1, 0, 0, 0], np.zeros(3), arm_override=override)
    assert np.allclose(out[rp.ARM_MOTORS], 0.4)


def test_projected_gravity_upright_and_tilted():
    assert np.allclose(rp.projected_gravity([1, 0, 0, 0]), [0, 0, -1])
    c, s = np.cos(np.pi / 8), np.sin(np.pi / 8)          # 45 deg pitch about y
    g = rp.projected_gravity([c, 0, s, 0])
    assert np.isclose(np.linalg.norm(g), 1.0) and abs(g[0]) > 0.6


def _controller():
    from rl_arm_controller import RlArmController

    ctl = RlArmController(log=lambda *a: None)
    start = np.zeros(29)
    ctl.latest = (start.copy(), np.zeros(29), np.array([1.0, 0, 0, 0]), np.zeros(3))
    ctl.on_first_state(start)
    ctl.ready.set()
    return ctl


def test_controller_phases_follow_rl_lab_walking_test():
    from rl_arm_controller import RAMP_S, SETTLE_S, STAND_KD, STAND_KP

    ctl = _controller()
    q, kp, kd, tau = ctl.build_command()
    assert ctl.phase() == 1 and np.allclose(kp, STAND_KP) and np.allclose(kd, STAND_KD)
    assert np.max(np.abs(q)) < 0.01                        # starts from where the robot is
    while ctl.t < RAMP_S + 0.5 * SETTLE_S:
        q, kp, kd, tau = ctl.build_command()
    assert ctl.phase() == 2 and np.allclose(q, rp.DEFAULT_POSE_MOTOR)
    while ctl.phase() < 3:
        ctl.build_command()
    q, kp, kd, tau = ctl.build_command()
    assert np.allclose(kp, rp.STIFFNESS) and np.allclose(kd, rp.DAMPING)
    assert np.all(tau[:15] == 0.0)                         # no feed-forward on the policy's joints


def test_controller_arms_only():
    ctl = _controller()
    with pytest.raises(ValueError):
        ctl.set_targets({12: 0.3})                         # waist belongs to the policy
    with pytest.raises(ValueError):
        ctl.set_targets({0: 0.3})                          # legs too
    ctl.set_targets({15: -0.5})
    while ctl.phase() < 3:
        ctl.build_command()
    for _ in range(400):
        q, *_ = ctl.build_command()
    assert q[15] == pytest.approx(-0.5, abs=0.02)          # arm reached its target in phase 3


def test_offline_policy_balances_while_arms_move():
    """unitree_mujoco's G1 on the floor, rl_lab policy balancing, one arm
    waving and the other going out to the side: the robot must stay up."""
    import mujoco as mj

    from command_shaping import CommandShaper
    from g1_sim import load_sim_model

    m = load_sim_model(fixed_base=False)
    m.opt.timestep = 0.005
    d = mj.MjData(m)
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(29)])
    dadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(29)])
    d.qpos[:] = m.qpos0
    d.qpos[qadr] = rp.DEFAULT_POSE_MOTOR
    mj.mj_forward(m, d)
    d.qpos[2] -= min(d.body(b).xpos[2] for b in ("left_ankle_roll_link", "right_ankle_roll_link")) - 0.045
    lo, hi = m.actuator_ctrlrange.T
    pol = rp.RlLabPolicy()
    arms = CommandShaper(rp.DEFAULT_POSE_MOTOR.copy(), 0.005, max_speed=1.0, engage_s=0.0)
    arms.set_target(16, 1.4)       # left arm out to the side
    arms.set_target(22, -1.2)      # right arm forward/up
    arms.set_target(25, 0.2)
    target = rp.DEFAULT_POSE_MOTOR.copy()
    for n in range(int(8.0 / 0.005)):
        if n % 4 == 0:
            target = pol.step(d.qpos[qadr], d.qvel[dadr], d.qpos[3:7], d.qvel[3:6],
                              arm_override={i: arms.q[i] for i in rp.ARM_MOTORS})
        arms.tick()
        target[rp.ARM_MOTORS] = arms.q[rp.ARM_MOTORS]
        d.ctrl[:] = np.clip(rp.STIFFNESS * (target - d.qpos[qadr]) - rp.DAMPING * d.qvel[dadr], lo, hi)
        mj.mj_step(m, d)
        assert d.qpos[2] > 0.6, f"fell at t={n * 0.005:.1f}s"
    assert d.qpos[qadr][16] == pytest.approx(1.4, abs=0.25)


def _arm_sdk_msg(weight, targets, kp=60.0, kd=1.5):
    from types import SimpleNamespace

    cmds = [SimpleNamespace(q=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(35)]
    for i, q in targets.items():
        cmds[i] = SimpleNamespace(q=q, kp=kp, kd=kd, tau=0.0)
    cmds[29] = SimpleNamespace(q=weight, kp=0.0, kd=0.0, tau=0.0)
    return SimpleNamespace(motor_cmd=cmds)


def test_onboard_balance_stands_and_blends_arm_sdk():
    """sim_standing.py --rl-lab without DDS: stand-up sequence, then the
    policy balances while rt/arm_sdk (weight 1) moves both arms."""
    import mujoco as mj

    from g1_sim import load_sim_model
    from sim_onboard import OnboardBalance

    m = load_sim_model(fixed_base=False)
    m.opt.timestep = 0.005
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    ob = OnboardBalance(m, d)
    qadr = ob.qadr
    lo, hi = m.actuator_ctrlrange.T
    arms = {i: float(rp.DEFAULT_POSE_MOTOR[i]) for i in rp.ARM_MOTORS}
    arms.update({16: 1.2, 23: -1.2, 18: 0.3, 25: 0.3})     # both arms out to the side, elbows bent
    torso = m.body("torso_link").id
    weight = float(m.body_subtreemass[1]) * 9.81
    for n in range(int(14.0 / 0.005)):
        t = n * 0.005
        # Band stand-in, as sim_standing.py --rl-lab: carries 80 % of the weight,
        # fades out 6-8 s (the policy takes over at 5 s).
        d.xfrc_applied[torso, 2] = 0.8 * weight * float(np.clip((8.0 - t) / 2.0, 0.0, 1.0))
        if t > 9.0:
            ob.on_arm_sdk(_arm_sdk_msg(min(1.0, (t - 9.0) / 2.0), arms))
        ob.control()
        d.ctrl[:] = np.clip(d.ctrl, lo, hi)
        mj.mj_step(m, d)
        if t > 8.0:
            assert d.qpos[2] > 0.6, f"fell at t={t:.1f}s"
    q = d.qpos[qadr]
    assert q[16] == pytest.approx(1.2, abs=0.2) and q[23] == pytest.approx(-1.2, abs=0.2)
    assert ob.policy_on
