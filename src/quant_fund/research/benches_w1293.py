"""Wave-1293 bench adapters: constitutional-AI canon (SYNTHETIC only)."""

from quant_fund.models import (
    cai_critique_studies,
    constitutional_studies,
    harmlessness_rl_studies,
    principle_eval_studies,
    rlaif_studies,
    sleeper_eval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12930


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cai_critique_studies_family(seed: int = _SEED + 0):
    """cai_critique_studies: synthetic correctness bench."""
    return _finite_blob(cai_critique_studies.bench_cai_critique_studies(seed))


def bench_constitutional_studies_family(seed: int = _SEED + 1):
    """constitutional_studies: synthetic correctness bench."""
    return _finite_blob(constitutional_studies.bench_constitutional_studies(seed))


def bench_harmlessness_rl_studies_family(seed: int = _SEED + 2):
    """harmlessness_rl_studies: synthetic correctness bench."""
    return _finite_blob(harmlessness_rl_studies.bench_harmlessness_rl_studies(seed))


def bench_principle_eval_studies_family(seed: int = _SEED + 3):
    """principle_eval_studies: synthetic correctness bench."""
    return _finite_blob(principle_eval_studies.bench_principle_eval_studies(seed))


def bench_rlaif_studies_family(seed: int = _SEED + 4):
    """rlaif_studies: synthetic correctness bench."""
    return _finite_blob(rlaif_studies.bench_rlaif_studies(seed))


def bench_sleeper_eval_studies_family(seed: int = _SEED + 5):
    """sleeper_eval_studies: synthetic correctness bench."""
    return _finite_blob(sleeper_eval_studies.bench_sleeper_eval_studies(seed))
