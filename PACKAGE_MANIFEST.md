# Package Manifest

This package is a public showcase extraction for AutoValiSearch.

## Included

- `README.md`: concise public project overview.
- `LICENSE`: Apache-2.0 license.
- `requirements.txt`: runtime dependency note.
- `docs/`: technical report and 5-page printable demo documents.
- `figures/`: workflow and result figures for README/docs.
- `agents/`, `baselines/`, `controller/`, `dataset/`, `llm/`, `models/`, `outer/`, `phase1_search/`, `phase2b_validation/`, `reporting/`, `training/`, `utils/`: the main code modules for the controlled workflow, search, validation, and training.
- `examples/`: trace/case studies, the training trial entry, and the stage workflow examples.
- `artifacts/`: sanitized sample logs, results, and configs.
- `src/`: workflow-oriented compatibility/demo code.

## Not Included

- private datasets;
- precomputed checkpoints or model weights;
- full experiment outputs;
- private backend URLs or API keys;
- local paths or environment files;
- experiment caches.

## Demo Scope

The included code is a public, runnable extraction of the controlled workflow. It includes the training trial entry, the formal stage suite entrypoints, the core search and validation code paths, and the stage workflow examples used to explain the system.
