from __future__ import annotations

from typing import Any, Iterable, Mapping

FORBIDDEN_PHRASES = [
    "AutoSOTA",
    "fully autonomous scientific discovery",
    "automatically discovers new architectures",
    "beats random best-of-64",
    "beats random best-of-k",
    "state-of-the-art autonomous scientist",
]

NEGATED_CLAIM_MARKERS = [
    "do not claim",
    "does not claim",
    "not claim",
    "must not",
    "should not",
    "not a formal",
    "not the formal",
    "not constitute",
    "not establish",
    "not used as",
    "without overclaim",
]


def _flatten_text(values: Iterable[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            parts.extend(str(item) for item in value.values() if isinstance(item, str))
        elif isinstance(value, Iterable):
            parts.extend(str(item) for item in value if isinstance(item, str))
    return "\n".join(parts)


def _is_negated_or_guardrail(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in NEGATED_CLAIM_MARKERS)


def _phrase_present_as_claim(combined: str, phrase: str) -> bool:
    phrase_lower = phrase.lower()
    for line in combined.splitlines():
        if phrase_lower in line.lower() and not _is_negated_or_guardrail(line):
            return True
    return False


def audit_claims(
    evidence_pack: Mapping[str, Any],
    *,
    report_texts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    allowed_claims = [str(item) for item in evidence_pack.get("allowed_claims", [])]
    not_allowed_claims = [str(item) for item in evidence_pack.get("not_allowed_claims", [])]
    run_metadata = evidence_pack.get("run_metadata", {}) if isinstance(evidence_pack.get("run_metadata", {}), Mapping) else {}
    texts = [str(value) for value in evidence_pack.values() if isinstance(value, str)]
    if report_texts:
        texts.extend(str(text) for text in report_texts.values())
    combined = "\n".join(texts)

    forbidden_detected = [phrase for phrase in FORBIDDEN_PHRASES if _phrase_present_as_claim(combined, phrase)]
    warnings = []
    if _phrase_present_as_claim(combined, "random best-of-k") or _phrase_present_as_claim(combined, "random best-of-64"):
        warnings.append("Random best-of-k appears in evidence pack and must not be used as the main comparison; best_test_upper_bound is the analysis upper bound.")
    if not bool(run_metadata.get("formal_performance_claims_allowed", False)) and ("outperforms random and tpe" in combined.lower() or "improves over standard vanilla best-val" in combined.lower()):
        warnings.append("Smoke run detected; formal performance claims must be suppressed.")

    passed = not forbidden_detected
    return {
        "allowed_claims_used": allowed_claims,
        "forbidden_claims_detected": forbidden_detected,
        "warnings": warnings,
        "passed": passed,
        "not_allowed_claims": not_allowed_claims,
    }
