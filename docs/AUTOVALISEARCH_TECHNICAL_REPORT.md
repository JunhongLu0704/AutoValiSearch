# AutoValiSearch Technical Report

AutoValiSearch is a controlled workflow for LLM-assisted experiment search and validation in vision training pipelines. The system is intentionally bounded: the LLM proposes structured candidates, while deterministic controllers validate and execute them.

## Architecture

Stage I searches training configurations in a fixed search space. One aggregate trial evaluates one configuration across 4 domain splits and 2 seeds. Stage II designs validation policies for checkpoint selection using cached validation views.

## Stage I Inputs And Outputs

Input: bounded parameter space, dataset splits, training template, LLM backend, and trial history.

Output: proposal JSON files, aggregate trial histories, best checkpoint pool, and summary tables.

## Stage II Inputs And Outputs

Input: Stage I best LLM checkpoints, validation policy DSL, augmentation registry, and prior policy feedback.

Output: policy traces, validation-view score caches, deployable policy summary, and analysis upper-bound metrics.

## Sample Results

| Dataset | Stage I LLM | Stage II Deployable | Gain Over Vanilla | Upper Bound |
|---|---:|---:|---:|---:|
| PACS | 77.226 | 77.736 | +0.510 | 78.310 |
| VLCS | 67.852 | 67.988 | +0.136 | 68.874 |

The upper bound uses test labels and is provided only for offline analysis.

## Runnable Training Code

The public package includes a real PyTorch training entry:

```bash
python scripts/run_trial.py --config path/to/config.json --trial_dir path/to/trial_dir
```

It uses the Stage I fields `lr`, `lambdap`, `epochp`, and `num_f`, writes epoch checkpoints, and exports `result.json` in the requested trial directory. The public repo also includes the formal stage suite entrypoints:

```bash
python scripts/run_phase1_formal_suite.py
python scripts/run_phase2b_formal_suite.py
```

The code path is visible, runnable, and organized around the same Stage I and Stage II abstractions used in the showcase workflow.

## Engineering Notes

The public package is code-forward. It includes the training trial entry, the formal stage suite entrypoints, and the controlled search/validation code paths used in the showcase workflow.
