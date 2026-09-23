"""The joint-name <-> motor-index mapping must agree with every model it is
used against: unitree_mujoco's MJCF (actuator i is motor i), GMR's G1
model (joints looked up by name) and the real-robot limit table."""

import mujoco
import pytest

from g1_joint_limits import G1_29DOF_JOINT_LIMITS
from g1_sim import SCENE_XML
from gmr_retarget import MOTOR_JOINT_NAMES


@pytest.fixture(scope="module")
def sim_model():
    return mujoco.MjModel.from_xml_path(SCENE_XML)


def test_sim_actuator_i_drives_motor_i(sim_model):
    assert sim_model.nu == 29
    for i in range(29):
        joint = sim_model.joint(sim_model.actuator_trnid[i, 0]).name
        assert joint == MOTOR_JOINT_NAMES[i], i


def test_sim_position_sensor_i_reads_motor_i(sim_model):
    # LowState motor_state[i].q = sensordata[i] in the sim bridge.
    for i in range(29):
        assert sim_model.sensor_objid[i] == sim_model.joint(MOTOR_JOINT_NAMES[i]).id


def test_limit_table_matches_sim_ranges(sim_model):
    for name, lim in G1_29DOF_JOINT_LIMITS.items():
        lo, hi = sim_model.joint(name).range
        assert lo == pytest.approx(lim.lower, abs=1e-3), name
        assert hi == pytest.approx(lim.upper, abs=1e-3), name


def test_gmr_model_has_every_joint_within_real_limits(pipeline):
    model = pipeline.gmr.model
    for name, lim in G1_29DOF_JOINT_LIMITS.items():
        lo, hi = model.joint(name).range
        # GMR's ranges are equal or tighter (G1_ARM_CONVENTIONS.md section 2).
        assert lo >= lim.lower - 1e-3 and hi <= lim.upper + 1e-3, name


def test_motor_names_unique_and_complete():
    assert len(set(MOTOR_JOINT_NAMES)) == 29
    assert set(MOTOR_JOINT_NAMES) == set(G1_29DOF_JOINT_LIMITS)
