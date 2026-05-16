from __future__ import annotations

from typing import Any, Dict, List


CAPABILITY_REGISTRY: List[Dict[str, Any]] = [
    {
        'name': 'design_validator_family',
        'owner': 'Val Agent',
        'input_schema': {
            'round_index': 'int',
            'fixed_run_context': 'object',
            'warmup_summary': 'object',
            'family_memory_view': 'object',
        },
        'output_schema': {'validator_family_recipe': 'object'},
        'side_effects': ['none'],
        'retryability': 'bounded_repair_retry',
        'execution_coupled': False,
    },
    {
        'name': 'repair_validator_family_recipe',
        'owner': 'Val Agent',
        'input_schema': {
            'previous_output': 'object_or_string',
            'validation_error_report': 'object',
        },
        'output_schema': {'validator_family_recipe': 'object'},
        'side_effects': ['none'],
        'retryability': 'bounded_repair_retry',
        'execution_coupled': False,
    },
    {
        'name': 'compile_validator_family_spec',
        'owner': 'Controller',
        'input_schema': {'validator_family_recipe': 'object'},
        'output_schema': {'validator_family_spec': 'object'},
        'side_effects': ['writes validator family artifacts'],
        'retryability': 'caller_owned',
        'execution_coupled': False,
    },
    {
        'name': 'summarize_family_feedback',
        'owner': 'Controller',
        'input_schema': {'family_round_records': 'list[object]', 'limits': 'object'},
        'output_schema': {'family_memory_view': 'object'},
        'side_effects': ['writes family memory artifacts'],
        'retryability': 'deterministic',
        'execution_coupled': False,
    },
    {
        'name': 'propose_training_config',
        'owner': 'Train Agent',
        'input_schema': {
            'locked_family_summary': 'object',
            'family_utility_summary': 'object',
            'best_trials': 'list[object]',
            'recent_failures': 'list[object]',
        },
        'output_schema': {'proposal': 'object'},
        'side_effects': ['none'],
        'retryability': 'caller_owned',
        'execution_coupled': False,
    },
    {
        'name': 'check_training_config_validity',
        'owner': 'Controller',
        'input_schema': {'proposal': 'object', 'search_space': 'object'},
        'output_schema': {'normalized_trial_request': 'object'},
        'side_effects': ['writes invalid proposal diagnostics on failure'],
        'retryability': 'caller_owned',
        'execution_coupled': False,
    },
    {
        'name': 'summarize_trial_history',
        'owner': 'Controller',
        'input_schema': {'trial_history': 'list[object]', 'top_k': 'int'},
        'output_schema': {'best_trials': 'list[object]', 'recent_failures': 'list[object]'},
        'side_effects': ['none'],
        'retryability': 'deterministic',
        'execution_coupled': False,
    },
    {
        'name': 'analyze_proxy_alignment',
        'owner': 'Controller',
        'input_schema': {'trial_pool': 'list[object]'},
        'output_schema': {'proxy_quality_summary': 'object'},
        'side_effects': ['writes project offline reports'],
        'retryability': 'deterministic',
        'execution_coupled': False,
    },
    {
        'name': 'summarize_benchmark_results',
        'owner': 'Controller',
        'input_schema': {'round_summaries': 'list[object]', 'budget_table': 'object'},
        'output_schema': {'benchmark_summary': 'object'},
        'side_effects': ['writes benchmark and leaderboard artifacts'],
        'retryability': 'deterministic',
        'execution_coupled': False,
    },
]


def build_capability_registry() -> Dict[str, Any]:
    return {
        'project_name': 'AutoValiSearch',
        'stage': 'Stage 2',
        'capabilities': CAPABILITY_REGISTRY,
    }
