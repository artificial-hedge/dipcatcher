"""Wave-1371 bench adapters: social-reasoning canon (SYNTHETIC only)."""

from quant_fund.models import (
    ethos_lite_studies,
    moral_stories_studies,
    mutual_lite_studies,
    prosocial_lite_studies,
    scruples_studies,
    siqa_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13710


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


def bench_ethos_lite_studies_family(seed: int = _SEED + 0):
    """ethos_lite_studies: synthetic correctness bench."""
    return _finite_blob(ethos_lite_studies.bench_ethos_lite_studies(seed))


def bench_moral_stories_studies_family(seed: int = _SEED + 1):
    """moral_stories_studies: synthetic correctness bench."""
    return _finite_blob(moral_stories_studies.bench_moral_stories_studies(seed))


def bench_mutual_lite_studies_family(seed: int = _SEED + 2):
    """mutual_lite_studies: synthetic correctness bench."""
    return _finite_blob(mutual_lite_studies.bench_mutual_lite_studies(seed))


def bench_prosocial_lite_studies_family(seed: int = _SEED + 3):
    """prosocial_lite_studies: synthetic correctness bench."""
    return _finite_blob(prosocial_lite_studies.bench_prosocial_lite_studies(seed))


def bench_scruples_studies_family(seed: int = _SEED + 4):
    """scruples_studies: synthetic correctness bench."""
    return _finite_blob(scruples_studies.bench_scruples_studies(seed))


def bench_siqa_lite_studies_family(seed: int = _SEED + 5):
    """siqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(siqa_lite_studies.bench_siqa_lite_studies(seed))
