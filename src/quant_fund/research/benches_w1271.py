"""Wave-1271 bench adapters: LLM-evaluation canon (SYNTHETIC only)."""

from quant_fund.models import (
    arena_battle_studies,
    bigbench_studies,
    capability_elicitation_studies,
    contamination_detect_studies,
    helm_eval_studies,
    llm_judge_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12710


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arena_battle_studies_family(seed: int = _SEED + 0):
    """arena_battle_studies: synthetic correctness bench."""
    return _finite_blob(arena_battle_studies.bench_arena_battle_studies(seed))


def bench_bigbench_studies_family(seed: int = _SEED + 1):
    """bigbench_studies: synthetic correctness bench."""
    return _finite_blob(bigbench_studies.bench_bigbench_studies(seed))


def bench_capability_elicitation_studies_family(seed: int = _SEED + 2):
    """capability_elicitation_studies: synthetic correctness bench."""
    return _finite_blob(capability_elicitation_studies.bench_capability_elicitation_studies(seed))


def bench_contamination_detect_studies_family(seed: int = _SEED + 3):
    """contamination_detect_studies: synthetic correctness bench."""
    return _finite_blob(contamination_detect_studies.bench_contamination_detect_studies(seed))


def bench_helm_eval_studies_family(seed: int = _SEED + 4):
    """helm_eval_studies: synthetic correctness bench."""
    return _finite_blob(helm_eval_studies.bench_helm_eval_studies(seed))


def bench_llm_judge_studies_family(seed: int = _SEED + 5):
    """llm_judge_studies: synthetic correctness bench."""
    return _finite_blob(llm_judge_studies.bench_llm_judge_studies(seed))
