"""Compatibility wrapper for Stage0 training (deprecated SP3 entrypoint)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("[DEPRECATED] examples/train_sp3_stage0_ppo.py now forwards to SP2 Stage0 target selection trainer.")

from whai.shooting.policies.sp2_targeting.stage0.train_ppo import main


if __name__ == "__main__":
    main()
