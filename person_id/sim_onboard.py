"""
"On-board" balance for the simulated G1: unitree_rl_lab's velocity policy
run inside the simulator, in simulated time, plus an rt/arm_sdk-style arm
override. It plays the part Unitree's locomotion firmware plays on the
real robot, so person_id drives the arms the same way in the sim as on the
G1: blend-weighted arm commands on rt/arm_sdk (arm_sdk_publisher.py).

Why inside the simulator rather than in a separate process, as the gesture
team's rl_lab_walking_test.py runs it: the policy was trained to act every
20 ms of simulated time on its last 5 observations. From another process it
acts every 20 ms of wall-clock time. On the team VM the simulator often runs
at 0.4-0.8x real time, and then the policy's timing no longer matches the
physics it controls: in our DDS tests the robot fell at random, sometimes
with the arms still. Stepping the policy with the physics removes that
dependency entirely: the policy logic itself (rl_lab_policy.py) is the same.

Sequence (simulated time, as in rl_lab_walking_test.py): 0-3 s ease to the
policy's default pose (kp 60, kd 1.5), 3-5 s hold, then the policy balances
with velocity command (0, 0, 0). rt/arm_sdk: motor_cmd[29].q is the weight
w in [0, 1]; for each arm joint the target, kp and kd are blended
(1 - w) * policy + w * arm_sdk, and arm_sdk's tau is scaled by w. Waist joints
are blended only if the message sets kp > 0 for them (person_id doesn't).
While w > 0 the policy is shown its arms at its default pose
(rl_lab_policy arm_obs="default").
"""

import numpy as np

from rl_lab_policy import ARM_MOTORS, DAMPING, DEFAULT_POSE_MOTOR, STEP_DT, STIFFNESS, RlLabPolicy

RAMP_S = 3.0
SETTLE_S = 2.0
STAND_KP, STAND_KD = 60.0, 1.5
WAIST_MOTORS = [12, 13, 14]


class OnboardBalance:
    def __init__(self, model, data):
        self.model, self.data = model, data
        nu = model.nu
        self.qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(nu)])
        self.dadr = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(nu)])
        self.decimation = max(1, round(STEP_DT / model.opt.timestep))
        self.policy = RlLabPolicy(arm_obs="default")
        self.t0 = data.time
        self.start_q = None
        self.steps = 0
        self.policy_targets = DEFAULT_POSE_MOTOR.copy()
        self.arm_cmd = None          # (weight, q, kp, kd, tau)
        self.policy_on = False

    def on_arm_sdk(self, msg):
        """rt/arm_sdk LowCmd_ callback (DDS thread)."""
        mc = msg.motor_cmd
        self.arm_cmd = (
            float(np.clip(mc[29].q, 0.0, 1.0)),
            np.array([mc[i].q for i in range(29)]), np.array([mc[i].kp for i in range(29)]),
            np.array([mc[i].kd for i in range(29)]), np.array([mc[i].tau for i in range(29)]),
        )

    def phase(self):
        t = self.data.time - self.t0
        return 1 if t < RAMP_S else 2 if t < RAMP_S + SETTLE_S else 3

    def control(self):
        """Set data.ctrl for the next physics step."""
        d = self.data
        q = d.qpos[self.qadr]
        dq = d.qvel[self.dadr]
        if self.start_q is None:
            self.start_q = q.copy()
        phase = self.phase()
        if phase < 3:
            ratio = min(1.0, (d.time - self.t0) / RAMP_S)
            q_des = (1 - ratio) * self.start_q + ratio * DEFAULT_POSE_MOTOR
            d.ctrl[:] = STAND_KP * (q_des - q) - STAND_KD * dq
            return
        self.policy_on = True
        w, q_sdk, kp_sdk, kd_sdk, tau_sdk = self.arm_cmd if self.arm_cmd is not None else (0.0, 0, 0, 0, 0)
        blended = [i for i in ARM_MOTORS] + ([j for j in WAIST_MOTORS if kp_sdk[j] > 0] if w > 0 else [])
        if self.steps % self.decimation == 0:
            override = None
            if w > 0:
                override = {i: float((1 - w) * self.policy_targets[i] + w * q_sdk[i]) for i in blended}
            quat = d.qpos[3:7]
            gyro = d.qvel[3:6]           # free-joint angular velocity is in the body frame
            self.policy_targets = self.policy.step(q, dq, quat, gyro, (0.0, 0.0, 0.0), override)
        self.steps += 1
        q_des, kp, kd, tau = self.policy_targets.copy(), STIFFNESS.copy(), DAMPING.copy(), np.zeros(29)
        if w > 0:
            b = np.array(blended)
            q_des[b] = (1 - w) * q_des[b] + w * q_sdk[b]
            kp[b] = (1 - w) * kp[b] + w * kp_sdk[b]
            kd[b] = (1 - w) * kd[b] + w * kd_sdk[b]
            tau[b] = w * tau_sdk[b]
        d.ctrl[:] = tau + kp * (q_des - q) - kd * dq
