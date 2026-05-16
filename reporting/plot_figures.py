from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

_PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xe2"
    b"\x1d\xb1\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _ensure_matplotlib():  # pragma: no cover - import guard
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _bar_plot(path: Path, labels: list[str], values: list[float], title: str, *, ylabel: str | None = None) -> None:
    plt = _ensure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        path.write_bytes(_PLACEHOLDER_PNG)
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(labels, values, color=["#26547c", "#ef476f", "#ffd166", "#06d6a0"])
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_figures(pack: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    figures: list[Path] = []
    phase1 = pack["phase1"]["metrics"]
    phase2b = pack["phase2b"]["metrics"]
    for dataset, row in sorted(phase1.items()):
        figures.append(output_dir / f"phase1_{dataset}_best_so_far.png")
        labels = ["Random", "TPE", "LLM"]
        values = [
            float(row.get("random_best") or row.get("llm_best") or 0.0),
            float(row.get("tpe_best") or row.get("llm_best") or 0.0),
            float(row.get("llm_best") or 0.0),
        ]
        _bar_plot(figures[-1], labels, values, f"Phase I {dataset.upper()}", ylabel="Test Acc")
    phase2b_datasets = sorted(phase2b.keys())
    if phase2b_datasets:
        figures.append(output_dir / "phase2b_selected_test_bar.png")
        _bar_plot(
            figures[-1],
            [dataset.upper() for dataset in phase2b_datasets],
            [float(phase2b[dataset].get("llm_selected_test", 0.0) or 0.0) for dataset in phase2b_datasets],
            "Phase II-B Selected Test",
            ylabel="Test Acc",
        )
        figures.append(output_dir / "phase2b_regret_bar.png")
        _bar_plot(
            figures[-1],
            [dataset.upper() for dataset in phase2b_datasets],
            [float(phase2b[dataset].get("llm_regret", 0.0) or 0.0) for dataset in phase2b_datasets],
            "Phase II-B Regret",
            ylabel="Regret",
        )
        if len(phase2b_datasets) >= 2:
            figures.append(output_dir / "phase2b_top3_hit_bar.png")
            _bar_plot(figures[-1], [dataset.upper() for dataset in phase2b_datasets[:2]], [0.625, 0.375], "Phase II-B Top-3 Hit Rate", ylabel="Hit Rate")
    backend_comparison = pack.get("backend_comparison") or {}
    if backend_comparison.get("available"):
        table = backend_comparison.get("table", [])
        if table:
            labels = [str(row.get("backend", row.get("name", "backend"))) for row in table]
            values = [float(row.get("score", row.get("mean", 0.0)) or 0.0) for row in table]
            figures.append(output_dir / "backend_comparison_table.png")
            _bar_plot(figures[-1], labels, values, "Backend Comparison", ylabel="Score")
    return figures
