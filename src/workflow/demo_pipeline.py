from pathlib import Path
import csv
import json


def summarize_artifacts(root):
    root = Path(root)
    phase1 = list(csv.DictReader((root / 'artifacts/sample_results/phase1_summary_table.csv').open('r', encoding='utf-8')))
    phase2 = json.loads((root / 'artifacts/sample_results/phase2b_summary.json').read_text(encoding='utf-8'))
    return {'phase1_rows': phase1, 'phase2_datasets': list(phase2['datasets'].keys())}
