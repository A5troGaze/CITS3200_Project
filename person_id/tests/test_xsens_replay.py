"""Optional Xsens path: GMR's bvh_xsens config on its own sample recording,
mapped to motor indices like the camera pipeline."""

import math
import os

import pytest

from g1_joint_limits import validate_command

SAMPLE = os.path.expanduser(
    "~/CITS3200/Dependencies/GMR/assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh")

pytestmark = pytest.mark.skipif(not os.path.exists(SAMPLE), reason="GMR sample BVH not present")


def test_xsens_sample_retargets_to_valid_upper_body_targets():
    from xsens_replay import XsensRetargeter, load_bvh

    frames, height, frame_time = load_bvh(SAMPLE, end=1200)
    assert 1.2 < height < 2.2
    assert frame_time == pytest.approx(1 / 120, rel=0.05)
    rt = XsensRetargeter(height)
    for frame in frames[::20]:
        targets, q, ms = rt.step(frame)
        assert set(targets) == set(range(12, 29))          # waist + arms, no legs
        assert all(math.isfinite(v) for v in targets.values())
        assert validate_command(q) == []
