from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIAL_DIR = ROOT / "outputs" / "training_trial_example"
TRIAL_DIR.mkdir(parents=True, exist_ok=True)

config = {
    "dataset": "PACS",
    "lr": 0.01,
    "lambdap": 1.0,
    "epochp": 2,
    "num_f": 3,
    "budget": "short",
    "bs": 64,
    "seed": 0,
    "bn_mode": "train",
    "workers": 0,
    "split_dir": str((ROOT / "splits" / "split_compositional_dominant_art_painting_target_sketch").resolve()),
    "image_root": str((ROOT.parents[2] / "data" / "PACS").resolve()),
}
config_path = TRIAL_DIR / "config.json"
config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "run_trial.py"),
    "--config",
    str(config_path),
    "--trial_dir",
    str(TRIAL_DIR),
]
env = os.environ.copy()
env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
subprocess.run(cmd, check=True, cwd=ROOT, env=env)

result = json.loads((TRIAL_DIR / "result.json").read_text(encoding="utf-8"))
print(json.dumps(
    {
        "status": result.get("status"),
        "best_val_epoch": result.get("best_val_epoch"),
        "best_selection_epoch": result.get("best_selection_epoch"),
        "best_val_acc1": result.get("best_val_acc1"),
        "best_test_acc1": result.get("best_test_acc1"),
        "selection_score": result.get("selection_score"),
        "trial_dir": str(TRIAL_DIR.relative_to(ROOT)),
    },
    indent=2,
    ensure_ascii=False,
))
