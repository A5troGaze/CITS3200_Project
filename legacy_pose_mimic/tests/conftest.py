import os
import sys

import pytest

# person_id's modules use bare imports (`from g1_joint_limits import ...`)
# assuming person_id/ itself is on sys.path, not its parent. Add it here
# so `pytest person_id/tests -q` works from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def pipeline():
    """One default MimicPipeline (anatomical, 3-DoF waist, no legs) shared
    by tests; building GMR takes ~0.5 s. Tests call pipeline.reset().

    GMR's per-frame time budget is raised so that convergence depends only
    on iteration counts, not on how busy the machine is; otherwise results
    change when the test run shares the CPU with the simulator. The latency
    test builds its own pipeline with the real budget."""
    from gmr_retarget import GmrRetargeter
    from mimic_pipeline import MimicPipeline

    return MimicPipeline(retargeter=GmrRetargeter(budget_ms=1000.0))
