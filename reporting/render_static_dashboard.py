from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any, Mapping

from utils.io import read_json, write_text
from utils.tables import html_table

from .plot_figures import build_figures


def _image_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.name}" />'


def render_dashboard(pack: Mapping[str, Any], report_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = build_figures(pack, output_dir / "figures")
    report_paths = {
        "demo_report": report_dir / "demo_report.md",
        "readme_summary": report_dir / "readme_summary.md",
        "resume_snippet": report_dir / "resume_snippet.md",
        "ppt_outline": report_dir / "ppt_outline.md",
        "limitations": report_dir / "limitations.md",
        "claim_audit": report_dir / "claim_audit.json",
    }
    phase2b_summary = pack.get("phase2b", {})
    claim_audit = read_json(report_dir / "claim_audit.json") if (report_dir / "claim_audit.json").exists() else {}
    val_trace_path = output_dir.parent / "phase2b" / "val_strategy_trace.md"
    val_trace = val_trace_path.read_text(encoding="utf-8") if val_trace_path.exists() else "No validation trace found."
    search_trace_path = output_dir.parent / "phase1" / "phase1_backend_trace.jsonl"
    search_trace_summary = search_trace_path.read_text(encoding="utf-8").splitlines()[:3] if search_trace_path.exists() else []
    backend_comparison = pack.get("backend_comparison", {})
    phase1_metrics = pack.get("phase1", {}).get("metrics", {})
    phase2b_metrics = pack.get("phase2b", {}).get("metrics", {})
    phase1_rows = [
        {
            "Dataset": dataset.upper(),
            "Random": f"{row.get('random_best', 0.0) or 0.0:.2f}" if row.get("random_best") is not None else "N/A",
            "TPE": f"{row.get('tpe_best', 0.0) or 0.0:.2f}" if row.get("tpe_best") is not None else "N/A",
            "LLM": f"{row.get('llm_best', 0.0) or 0.0:.2f}",
        }
        for dataset, row in sorted(phase1_metrics.items())
    ]
    run_metadata = pack.get("run_metadata", {})
    phase2b_rows = [
        {
            "Dataset": dataset.upper(),
            "Final epoch": f"{row.get('final_epoch', 0.0) or 0.0:.2f}",
            "Vanilla": f"{row.get('vanilla_selected_test', 0.0) or 0.0:.2f}",
            "Best-test upper bound": f"{row.get('best_test_upper_bound', row.get('vanilla_selected_test', 0.0)) or 0.0:.2f}",
            "Handcrafted": f"{row.get('handcrafted_photometric_mean_minus_std', 0.0) or 0.0:.2f}",
            "LLM-designed": f"{row.get('llm_selected_test', 0.0) or 0.0:.2f}",
            "Controller-safe": f"{row.get('controller_safe_validator', row.get('vanilla_selected_test', 0.0)) or 0.0:.2f}",
        }
        for dataset, row in sorted(phase2b_metrics.items())
    ]

    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>AutoValiSearch Dashboard</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1180px;margin:0 auto;padding:24px;line-height:1.5;color:#222}h1,h2{margin-bottom:0.35em}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{border:1px solid #ddd;border-radius:8px;padding:16px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #eee;padding:8px;text-align:left}img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}.muted{color:#666}.trace{font-family:Consolas,monospace;font-size:13px;white-space:pre-wrap}.pill{display:inline-block;border:1px solid #bbb;border-radius:999px;padding:2px 8px;margin:2px;background:#fafafa}</style>",
        "</head><body>",
        "<h1>AutoValiSearch</h1>",
        "<p class='muted'>Evidence dashboard for Phase I, Phase II-B, claim audit, and backend comparison.</p>",
        "<p class='muted'>Phase II-B uses a controlled validation policy interface: LLM-designed views and selection rules are validated, cached with per-round policy traces, evaluated, and protected by controller safety fallback.</p>",
        "<div class='grid'>",
        "<div class='card'><h2>Overview</h2>",
        f"<p>Backend: {pack['architecture'].get('backend', 'OpenAI-compatible cloud/local LLM backend')}</p>",
        f"<p>Agents: {', '.join(item if isinstance(item, str) else str(item.get('name', '')) for item in pack['architecture'].get('agents', []))}</p>",
        f"<p>Run type: {run_metadata.get('run_type', 'formal')}</p>",
        f"<p>Formal claims allowed: {run_metadata.get('formal_performance_claims_allowed', False)}</p>",
        "</div>",
        "<div class='card'><h2>Claim audit</h2>",
        f"<p>Passed: {claim_audit.get('passed', False)}</p>",
        f"<p>Warnings: {', '.join(claim_audit.get('warnings', []) or ['none'])}</p>",
        f"<p>Forbidden detected: {', '.join(claim_audit.get('forbidden_claims_detected', []) or ['none'])}</p>",
        "</div>",
        "<div class='card'><h2>Phase I</h2>",
        html_table(["Dataset", "Random", "TPE", "LLM"], phase1_rows or [{"Dataset": "N/A", "Random": "N/A", "TPE": "N/A", "LLM": "N/A"}]),
        "</div>",
        "<div class='card'><h2>Phase II-B</h2>",
        html_table(["Dataset", "Final epoch", "Vanilla", "Best-test upper bound", "Handcrafted", "LLM-designed", "Controller-safe"], phase2b_rows or [{"Dataset": "N/A", "Final epoch": "N/A", "Vanilla": "N/A", "Best-test upper bound": "N/A", "Handcrafted": "N/A", "LLM-designed": "N/A", "Controller-safe": "N/A"}]),
        "</div>",
        "<div class='card'><h2>Backend comparison</h2>",
        f"<p>Available: {bool(backend_comparison.get('available', False))}</p>",
        f"<p>Summary path: {backend_comparison.get('summary_path', 'n/a')}</p>",
        "</div>",
        "<div class='card'><h2>Phase II-B trace</h2>",
        f"<div class='trace'>{val_trace}</div>",
        "</div>",
        "<div class='card'><h2>Search trace</h2>",
        f"<div class='trace'>{'<br>'.join(search_trace_summary) if search_trace_summary else 'No search trace found.'}</div>",
        "</div>",
        "</div>",
        "<div class='grid'>",
    ]
    for figure in figures:
        html.append(f"<div class='card'>{_image_tag(figure)}</div>")
    html.append("</div>")
    html.append("<div class='card'><h2>Evidence Reporter outputs</h2><ul>")
    for name, path in report_paths.items():
        html.append(f"<li>{name}: {path.name}</li>")
    html.append("</ul></div>")
    html.append("<div class='card'><h2>Artifacts</h2><p>Evidence pack, claim audit, report outputs, dashboard figures, and static index.html are written under the formal output directory.</p></div>")
    html.append("</body></html>")
    index_path = output_dir / "index.html"
    write_text(index_path, "".join(html))
    return index_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the AutoValiSearch static dashboard")
    parser.add_argument("--evidence-pack", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pack = read_json(args.evidence_pack)
    render_dashboard(pack, Path(args.report_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
