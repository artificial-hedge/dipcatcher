"""Wave-1273 bench adapters: AI-safety canon (SYNTHETIC only)."""

from quant_fund.models import (
    alignment_eval_studies,
    guardrail_studies,
    hallucination_detect_studies,
    jailbreak_defense_studies,
    red_team_studies,
    sleeper_agent_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12730


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alignment_eval_studies_family(seed: int = _SEED + 0):
    """alignment_eval_studies: synthetic correctness bench."""
    return _finite_blob(alignment_eval_studies.bench_alignment_eval_studies(seed))


def bench_guardrail_studies_family(seed: int = _SEED + 1):
    """guardrail_studies: synthetic correctness bench."""
    return _finite_blob(guardrail_studies.bench_guardrail_studies(seed))


def bench_hallucination_detect_studies_family(seed: int = _SEED + 2):
    """hallucination_detect_studies: synthetic correctness bench."""
    return _finite_blob(hallucination_detect_studies.bench_hallucination_detect_studies(seed))


def bench_jailbreak_defense_studies_family(seed: int = _SEED + 3):
    """jailbreak_defense_studies: synthetic correctness bench."""
    return _finite_blob(jailbreak_defense_studies.bench_jailbreak_defense_studies(seed))


def bench_red_team_studies_family(seed: int = _SEED + 4):
    """red_team_studies: synthetic correctness bench."""
    return _finite_blob(red_team_studies.bench_red_team_studies(seed))


def bench_sleeper_agent_studies_family(seed: int = _SEED + 5):
    """sleeper_agent_studies: synthetic correctness bench."""
    return _finite_blob(sleeper_agent_studies.bench_sleeper_agent_studies(seed))
