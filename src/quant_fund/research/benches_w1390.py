"""Wave-1390 bench adapters: multi-doc-sum canon (SYNTHETIC only)."""

from quant_fund.models import (
    episum_lite_studies,
    fsum_lite_studies,
    mds_news_studies,
    sqcs_lite_studies,
    summon_fce_studies,
    wcep_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13900


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


def bench_episum_lite_studies_family(seed: int = _SEED + 0):
    """episum_lite_studies: synthetic correctness bench."""
    return _finite_blob(episum_lite_studies.bench_episum_lite_studies(seed))


def bench_fsum_lite_studies_family(seed: int = _SEED + 1):
    """fsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(fsum_lite_studies.bench_fsum_lite_studies(seed))


def bench_mds_news_studies_family(seed: int = _SEED + 2):
    """mds_news_studies: synthetic correctness bench."""
    return _finite_blob(mds_news_studies.bench_mds_news_studies(seed))


def bench_sqcs_lite_studies_family(seed: int = _SEED + 3):
    """sqcs_lite_studies: synthetic correctness bench."""
    return _finite_blob(sqcs_lite_studies.bench_sqcs_lite_studies(seed))


def bench_summon_fce_studies_family(seed: int = _SEED + 4):
    """summon_fce_studies: synthetic correctness bench."""
    return _finite_blob(summon_fce_studies.bench_summon_fce_studies(seed))


def bench_wcep_lite_studies_family(seed: int = _SEED + 5):
    """wcep_lite_studies: synthetic correctness bench."""
    return _finite_blob(wcep_lite_studies.bench_wcep_lite_studies(seed))
