from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.inner_loop import (
    ConfigError,
    TrialFailure,
    compute_config_hash,
    compute_fallback_config_hash,
    normalize_config,
    run_inner_loop,
    validate_config,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')


def _cleanup_failed_trial_checkpoints(trial_dir: Path) -> dict:
    removed = []
    for path in trial_dir.rglob('*.pth'):
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed.append(str(path.relative_to(trial_dir)))
        except FileNotFoundError:
            continue
    return {
        'removed_checkpoint_count': len(removed),
        'removed_checkpoint_paths': removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run one AutoValiSearch inner-loop trial')
    parser.add_argument('--config', required=True, type=str, help='path to a trial JSON config')
    parser.add_argument('--trial_dir', required=True, type=str, help='directory to store trial artifacts')
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    result_path = trial_dir / 'result.json'
    error_path = trial_dir / 'error.txt'
    normalized_config = None
    config_hash = None

    try:
        raw_config = json.loads(config_path.read_text(encoding='utf-8'))
        normalized_config = normalize_config(raw_config, str(trial_dir))
        config_hash = compute_config_hash(normalized_config)
        normalized_config['config_hash'] = config_hash
        _write_json(trial_dir / 'config.json', dict(normalized_config))
        validate_config(normalized_config)

        start_time = __import__('time').time()
        result = run_inner_loop(normalized_config, str(trial_dir))
        result['runtime_sec'] = round(__import__('time').time() - start_time, 4)
        _write_json(result_path, result)
    except Exception as exc:
        if normalized_config is None:
            try:
                raw_config = json.loads(config_path.read_text(encoding='utf-8'))
            except Exception:
                raw_config = {'config_path': str(config_path)}
            config_hash = compute_fallback_config_hash(raw_config)
        else:
            config_hash = normalized_config.get('config_hash', config_hash)

        traceback_text = traceback.format_exc()
        error_path.write_text(traceback_text, encoding='utf-8')

        if isinstance(exc, TrialFailure):
            fail_reason = exc.fail_reason
        elif isinstance(exc, ConfigError):
            fail_reason = 'invalid_config'
        else:
            fail_reason = 'unhandled_exception'

        cleanup_summary = _cleanup_failed_trial_checkpoints(trial_dir)
        failure_result = {
            'status': 'fail',
            'config_hash': config_hash,
            'fail_reason': fail_reason,
            'error_type': exc.__class__.__name__,
            'error_message': str(exc),
            'traceback_path': os.path.relpath(error_path, trial_dir),
            **cleanup_summary,
        }
        _write_json(result_path, failure_result)
        if isinstance(exc, (TrialFailure, ConfigError)):
            print(f'[FAIL] {fail_reason}: {exc}. See {error_path}', file=sys.stderr)
            raise SystemExit(2)
        raise


if __name__ == '__main__':
    main()

