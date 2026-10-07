"""Wave-1348 bench adapters: NLI-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    anli_r1_studies,
    anli_r2_studies,
    anli_r3_studies,
    mnli_match_studies,
    scitail_lite_studies,
    snli_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13480


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


def bench_anli_r1_studies_family(seed: int = _SEED + 0):
    """anli_r1_studies: synthetic correctness bench."""
    return _finite_blob(anli_r1_studies.bench_anli_r1_studies(seed))


def bench_anli_r2_studies_family(seed: int = _SEED + 1):
    """anli_r2_studies: synthetic correctness bench."""
    return _finite_blob(anli_r2_studies.bench_anli_r2_studies(seed))


def bench_anli_r3_studies_family(seed: int = _SEED + 2):
    """anli_r3_studies: synthetic correctness bench."""
    return _finite_blob(anli_r3_studies.bench_anli_r3_studies(seed))


def bench_mnli_match_studies_family(seed: int = _SEED + 3):
    """mnli_match_studies: synthetic correctness bench."""
    return _finite_blob(mnli_match_studies.bench_mnli_match_studies(seed))


def bench_scitail_lite_studies_family(seed: int = _SEED + 4):
    """scitail_lite_studies: synthetic correctness bench."""
    return _finite_blob(scitail_lite_studies.bench_scitail_lite_studies(seed))


def bench_snli_lite_studies_family(seed: int = _SEED + 5):
    """snli_lite_studies: synthetic correctness bench."""
    return _finite_blob(snli_lite_studies.bench_snli_lite_studies(seed))
