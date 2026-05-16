from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outer.controller import AutoValiSearchController, StopConfig
from outer.schema import default_search_space


def _load_json_optional(path: str | None):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run AutoValiSearch Stage 2 controller')
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split_dir', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--budget', default='medium')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_split_dirs', nargs='*', default=None)
    parser.add_argument('--eval_seeds', nargs='*', type=int, default=None)
    parser.add_argument('--per_split_parallel_evals', type=int, default=2)
    parser.add_argument('--bn_mode', default='eval')
    parser.add_argument('--amp', default='1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--rsw_min', type=float, default=0.2)
    parser.add_argument('--validator_protocol', default='validator_family')
    parser.add_argument('--validator_preset', default='llm_family_v1')
    parser.add_argument('--validator_family_spec_path', default=None)
    parser.add_argument('--validator_spec_path', default=None)
    parser.add_argument('--aggregate_objective', default='j_trial')
    parser.add_argument('--warmup_probe_count', type=int, default=3)
    parser.add_argument('--u_family_gamma', type=float, default=0.5)
    parser.add_argument('--design_retry_limit', type=int, default=3)
    parser.add_argument('--max_family_rounds', type=int, default=1)
    parser.add_argument('--family_memory_top_k', type=int, default=3)
    parser.add_argument('--family_memory_failure_k', type=int, default=3)
    parser.add_argument('--agent_mode', default='heuristic')
    parser.add_argument('--agent_model', default='gpt-5.4')
    parser.add_argument('--agent_base_url', default='https://www.autodl.art/api/v1')
    parser.add_argument('--agent_api_key_env', default='AUTODL_API_KEY')
    parser.add_argument('--agent_timeout_sec', type=float, default=180.0)
    parser.add_argument('--agent_max_attempts', type=int, default=4)
    parser.add_argument('--agent_backend', default=None)
    parser.add_argument('--agent_server_url', default=None)
    parser.add_argument('--agent_model_name', default=None)
    parser.add_argument('--agent_max_retries', type=int, default=None)
    parser.add_argument('--val_agent_mode', default=None)
    parser.add_argument('--val_agent_model', default=None)
    parser.add_argument('--val_agent_base_url', default=None)
    parser.add_argument('--val_agent_api_key_env', default=None)
    parser.add_argument('--val_agent_timeout_sec', type=float, default=None)
    parser.add_argument('--val_agent_max_attempts', type=int, default=None)
    parser.add_argument('--val_agent_backend', default=None)
    parser.add_argument('--val_agent_server_url', default=None)
    parser.add_argument('--val_agent_model_name', default=None)
    parser.add_argument('--val_agent_max_retries', type=int, default=None)
    parser.add_argument('--train_agent_mode', default=None)
    parser.add_argument('--train_agent_model', default=None)
    parser.add_argument('--train_agent_base_url', default=None)
    parser.add_argument('--train_agent_api_key_env', default=None)
    parser.add_argument('--train_agent_timeout_sec', type=float, default=None)
    parser.add_argument('--train_agent_max_attempts', type=int, default=None)
    parser.add_argument('--train_agent_backend', default=None)
    parser.add_argument('--train_agent_server_url', default=None)
    parser.add_argument('--train_agent_model_name', default=None)
    parser.add_argument('--train_agent_max_retries', type=int, default=None)
    parser.add_argument('--python_executable', default=sys.executable)
    parser.add_argument('--max_trials', type=int, default=20)
    parser.add_argument('--max_failures', type=int, default=6)
    parser.add_argument('--max_runtime_hours', type=float, default=8.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--design_only', action='store_true')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fixed_config = {
        'dataset': args.dataset,
        'split_dir': args.split_dir,
        'image_root': args.image_root,
        'budget': args.budget,
        'seed': args.seed,
        'eval_split_dirs': args.eval_split_dirs,
        'eval_seeds': args.eval_seeds,
        'per_split_parallel_evals': args.per_split_parallel_evals,
        'bn_mode': args.bn_mode,
        'amp': args.amp,
        'workers': args.workers,
        'prefetch_factor': args.prefetch_factor,
        'bs': args.bs,
        'rsw_min': args.rsw_min,
        'validator_protocol': args.validator_protocol,
        'validator_preset': args.validator_preset,
        'validator_family_spec': _load_json_optional(args.validator_family_spec_path),
        'validator_spec': _load_json_optional(args.validator_spec_path),
        'aggregate_objective': args.aggregate_objective,
        'warmup_probe_count': args.warmup_probe_count,
        'u_family_gamma': args.u_family_gamma,
        'design_retry_limit': args.design_retry_limit,
        'max_family_rounds': args.max_family_rounds,
        'family_memory_top_k': args.family_memory_top_k,
        'family_memory_failure_k': args.family_memory_failure_k,
        'disturb_mode': 'rsw',
    }
    search_space = default_search_space(fixed_config)
    stop_config = StopConfig(
        max_trials=args.max_trials,
        max_failures=args.max_failures,
        max_runtime_hours=args.max_runtime_hours,
        patience=args.patience,
    )
    controller = AutoValiSearchController(
        workdir=args.workdir,
        search_space=search_space,
        stop_config=stop_config,
        agent_mode=args.agent_mode,
        agent_model=args.agent_model,
        agent_base_url=args.agent_base_url,
        agent_api_key_env=args.agent_api_key_env,
        agent_timeout_sec=args.agent_timeout_sec,
        agent_max_attempts=args.agent_max_attempts,
        agent_backend=args.agent_backend,
        agent_server_url=args.agent_server_url,
        agent_model_name=args.agent_model_name,
        agent_max_retries=args.agent_max_retries,
        val_agent_mode=args.val_agent_mode,
        val_agent_model=args.val_agent_model,
        val_agent_base_url=args.val_agent_base_url,
        val_agent_api_key_env=args.val_agent_api_key_env,
        val_agent_timeout_sec=args.val_agent_timeout_sec,
        val_agent_max_attempts=args.val_agent_max_attempts,
        val_agent_backend=args.val_agent_backend,
        val_agent_server_url=args.val_agent_server_url,
        val_agent_model_name=args.val_agent_model_name,
        val_agent_max_retries=args.val_agent_max_retries,
        train_agent_mode=args.train_agent_mode,
        train_agent_model=args.train_agent_model,
        train_agent_base_url=args.train_agent_base_url,
        train_agent_api_key_env=args.train_agent_api_key_env,
        train_agent_timeout_sec=args.train_agent_timeout_sec,
        train_agent_max_attempts=args.train_agent_max_attempts,
        train_agent_backend=args.train_agent_backend,
        train_agent_server_url=args.train_agent_server_url,
        train_agent_model_name=args.train_agent_model_name,
        train_agent_max_retries=args.train_agent_max_retries,
        python_executable=args.python_executable,
        design_only=args.design_only,
    )
    summary = controller.run()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()

