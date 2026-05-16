from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.backend_config import load_backend_choice
from agents.search_agent import SearchAgent
from utils.io import write_json, write_jsonl, write_text
from .llm_search import propose_llm
from .botorch_search import propose_botorch
from .random_search import propose_random
from .replay_evaluator import evaluate_search_proposal
from .search_memory import summarize_search_memory
from .tpe_search import propose_tpe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a demo Phase I search agent")
    parser.add_argument("--strategy", choices=["random", "tpe", "botorch", "llm"], default="llm")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend-config")
    parser.add_argument("--backend-name")
    parser.add_argument("--backend", choices=["cloud", "local_openai_compatible"])
    parser.add_argument("--use-example-artifacts", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = None
    if args.backend_config:
        backend = load_backend_choice(args.backend_config, args.backend_name)
    agent = SearchAgent(backend=backend, strategy=args.strategy, seed=0)
    proposals = []
    trace_rows = []
    history = []
    for round_index in range(int(args.count)):
        if backend is not None:
            round_proposals = agent.propose(count=1, history=history)
            if not round_proposals:
                break
            proposal = round_proposals[0]
        elif args.strategy == "random":
            round_proposals = propose_random(count=1, seed=round_index, history=history)
            if not round_proposals:
                break
            proposal = round_proposals[0]
        elif args.strategy == "tpe":
            round_proposals = propose_tpe(count=1, history=history)
            if not round_proposals:
                break
            proposal = round_proposals[0]
        elif args.strategy == "botorch":
            round_proposals = propose_botorch(count=1, history=history)
            if not round_proposals:
                break
            proposal = round_proposals[0]
        else:
            round_proposals = propose_llm(count=1, history=history)
            if not round_proposals:
                break
            proposal = round_proposals[0]
        result = evaluate_search_proposal(proposal, history, round_index=round_index + 1)
        proposals.append(proposal)
        history.append(result)
        trace_rows.append(
            {
                **result,
                "backend_name": backend.name if backend else (args.backend or "deterministic_fallback"),
                "model": backend.model if backend else "offline-replay",
                "base_url_redacted": backend.base_url if backend else None,
                "latency_sec": 0.0,
                "json_valid": True,
                "fallback_used": backend is None,
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_search_memory(history, budget=int(args.count))
    write_json(output_dir / "phase1_search_summary.json", {"strategy": args.strategy, "proposal_count": len(proposals), "backend": backend.public_dict() if backend else None})
    write_json(output_dir / "phase1_search_proposals.json", proposals)
    write_jsonl(output_dir / "search_trace.jsonl", trace_rows)
    write_json(output_dir / "search_memory_summary.json", summary)
    write_text(
        output_dir / "search_agent_report.md",
        "\n".join(
            [
                "# Search Agent Report",
                "",
                f"- Strategy: {args.strategy}",
                f"- Evaluated trials: {len(history)}",
                f"- Current best: {summary.get('current_best')}",
                f"- Recent trend: {summary.get('recent_trend')}",
            ]
        ),
    )
    print(json.dumps({"strategy": args.strategy, "proposal_count": len(proposals), "best": summary.get("current_best")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

