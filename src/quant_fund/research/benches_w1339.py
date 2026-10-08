"""Wave-1339 bench adapters: math-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    aqua_rat_studies,
    geo_qa_studies,
    hol_step_studies,
    math_odyssey_studies,
    tab_math_studies,
    uni_math_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13390


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


def bench_aqua_rat_studies_family(seed: int = _SEED + 0):
    """aqua_rat_studies: synthetic correctness bench."""
    return _finite_blob(aqua_rat_studies.bench_aqua_rat_studies(seed))


def bench_geo_qa_studies_family(seed: int = _SEED + 1):
    """geo_qa_studies: synthetic correctness bench."""
    return _finite_blob(geo_qa_studies.bench_geo_qa_studies(seed))


def bench_hol_step_studies_family(seed: int = _SEED + 2):
    """hol_step_studies: synthetic correctness bench."""
    return _finite_blob(hol_step_studies.bench_hol_step_studies(seed))


def bench_math_odyssey_studies_family(seed: int = _SEED + 3):
    """math_odyssey_studies: synthetic correctness bench."""
    return _finite_blob(math_odyssey_studies.bench_math_odyssey_studies(seed))


def bench_tab_math_studies_family(seed: int = _SEED + 4):
    """tab_math_studies: synthetic correctness bench."""
    return _finite_blob(tab_math_studies.bench_tab_math_studies(seed))


def bench_uni_math_studies_family(seed: int = _SEED + 5):
    """uni_math_studies: synthetic correctness bench."""
    return _finite_blob(uni_math_studies.bench_uni_math_studies(seed))
