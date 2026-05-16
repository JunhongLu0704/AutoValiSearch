from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an OpenAI-compatible backend config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="LOCAL_LLM_API_KEY")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_backend": args.backend_name,
        "backends": {
            args.backend_name: {
                "base_url": args.base_url,
                "api_key_env": args.api_key_env,
                "model": args.model,
                "timeout_sec": int(args.timeout_sec),
                "max_retries": int(args.max_retries),
            }
        },
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
