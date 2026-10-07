"""Wave-1347 bench adapters: logical-reasoning-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    abductive_nli_studies,
    conseq_log_studies,
    logiqa_log_studies,
    lsat_log_studies,
    reason_mc_studies,
    recli_log_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13470


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


def bench_abductive_nli_studies_family(seed: int = _SEED + 0):
    """abductive_nli_studies: synthetic correctness bench."""
    return _finite_blob(abductive_nli_studies.bench_abductive_nli_studies(seed))


def bench_conseq_log_studies_family(seed: int = _SEED + 1):
    """conseq_log_studies: synthetic correctness bench."""
    return _finite_blob(conseq_log_studies.bench_conseq_log_studies(seed))


def bench_logiqa_log_studies_family(seed: int = _SEED + 2):
    """logiqa_log_studies: synthetic correctness bench."""
    return _finite_blob(logiqa_log_studies.bench_logiqa_log_studies(seed))


def bench_lsat_log_studies_family(seed: int = _SEED + 3):
    """lsat_log_studies: synthetic correctness bench."""
    return _finite_blob(lsat_log_studies.bench_lsat_log_studies(seed))


def bench_reason_mc_studies_family(seed: int = _SEED + 4):
    """reason_mc_studies: synthetic correctness bench."""
    return _finite_blob(reason_mc_studies.bench_reason_mc_studies(seed))


def bench_recli_log_studies_family(seed: int = _SEED + 5):
    """recli_log_studies: synthetic correctness bench."""
    return _finite_blob(recli_log_studies.bench_recli_log_studies(seed))
