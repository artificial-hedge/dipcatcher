"""Wave-1327 bench adapters: safety-bias-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    bbq_bias_studies,
    bold_bias_studies,
    crowspairs_studies,
    holist_bias_studies,
    realtoxicity_studies,
    toxigen_eval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13270


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


def bench_bbq_bias_studies_family(seed: int = _SEED + 0):
    """bbq_bias_studies: synthetic correctness bench."""
    return _finite_blob(bbq_bias_studies.bench_bbq_bias_studies(seed))


def bench_bold_bias_studies_family(seed: int = _SEED + 1):
    """bold_bias_studies: synthetic correctness bench."""
    return _finite_blob(bold_bias_studies.bench_bold_bias_studies(seed))


def bench_crowspairs_studies_family(seed: int = _SEED + 2):
    """crowspairs_studies: synthetic correctness bench."""
    return _finite_blob(crowspairs_studies.bench_crowspairs_studies(seed))


def bench_holist_bias_studies_family(seed: int = _SEED + 3):
    """holist_bias_studies: synthetic correctness bench."""
    return _finite_blob(holist_bias_studies.bench_holist_bias_studies(seed))


def bench_realtoxicity_studies_family(seed: int = _SEED + 4):
    """realtoxicity_studies: synthetic correctness bench."""
    return _finite_blob(realtoxicity_studies.bench_realtoxicity_studies(seed))


def bench_toxigen_eval_studies_family(seed: int = _SEED + 5):
    """toxigen_eval_studies: synthetic correctness bench."""
    return _finite_blob(toxigen_eval_studies.bench_toxigen_eval_studies(seed))
