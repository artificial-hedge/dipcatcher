"""Wave-1369 bench adapters: math-word-problem canon (SYNTHETIC only)."""

from quant_fund.models import (
    aime_eval_studies,
    asdiv_lite_studies,
    math500_lite_studies,
    mgsm_lite_studies,
    minerva_math_studies,
    svamp_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13690


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


def bench_aime_eval_studies_family(seed: int = _SEED + 0):
    """aime_eval_studies: synthetic correctness bench."""
    return _finite_blob(aime_eval_studies.bench_aime_eval_studies(seed))


def bench_asdiv_lite_studies_family(seed: int = _SEED + 1):
    """asdiv_lite_studies: synthetic correctness bench."""
    return _finite_blob(asdiv_lite_studies.bench_asdiv_lite_studies(seed))


def bench_math500_lite_studies_family(seed: int = _SEED + 2):
    """math500_lite_studies: synthetic correctness bench."""
    return _finite_blob(math500_lite_studies.bench_math500_lite_studies(seed))


def bench_mgsm_lite_studies_family(seed: int = _SEED + 3):
    """mgsm_lite_studies: synthetic correctness bench."""
    return _finite_blob(mgsm_lite_studies.bench_mgsm_lite_studies(seed))


def bench_minerva_math_studies_family(seed: int = _SEED + 4):
    """minerva_math_studies: synthetic correctness bench."""
    return _finite_blob(minerva_math_studies.bench_minerva_math_studies(seed))


def bench_svamp_lite_studies_family(seed: int = _SEED + 5):
    """svamp_lite_studies: synthetic correctness bench."""
    return _finite_blob(svamp_lite_studies.bench_svamp_lite_studies(seed))
