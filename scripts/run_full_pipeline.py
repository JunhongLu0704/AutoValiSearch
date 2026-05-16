from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd1 = [sys.executable, str(ROOT / "scripts" / "run_phase1_formal_suite.py")]
    cmd2 = [sys.executable, str(ROOT / "scripts" / "run_phase2b_formal_suite.py")]
    subprocess.run(cmd1, check=True, cwd=ROOT)
    subprocess.run(cmd2, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

