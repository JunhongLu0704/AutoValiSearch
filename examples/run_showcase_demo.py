from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
phase1_path = ROOT / 'artifacts' / 'sample_results' / 'phase1_summary_table.csv'
phase2_path = ROOT / 'artifacts' / 'sample_results' / 'phase2b_summary.json'

print('AutoValiSearch lightweight artifact demo')
print('=' * 48)

print('\nStage I summary')
with phase1_path.open('r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        print(f"{row['dataset']:4s} {row['method']:14s} best={float(row['best_mean_test_acc']):.3f} trials={row['trial_count']}")

print('\nStage II summary')
phase2 = json.loads(phase2_path.read_text(encoding='utf-8'))
for dataset, data in phase2['datasets'].items():
    best = data['best_llm_protocol']
    print(
        f"{dataset:4s} policy={best['policy_name']} "
        f"selected={best['deployable_selected_test_mean']:.3f} "
        f"gain={best['deployable_improvement_over_vanilla']:.3f}"
    )

print('\nThis demo reads sample artifacts only; it does not run model training.')
