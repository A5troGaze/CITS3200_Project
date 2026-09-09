import os
import sys

# person_id's modules use bare imports (`from g1_joint_limits import ...`)
# assuming person_id/ itself is on sys.path, not its parent. Add it here
# so `pytest person_id/tests -q` works from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
