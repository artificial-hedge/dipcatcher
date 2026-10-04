"""Wave-1383 bench adapters: frontier-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    aime24_studies,
    gpqa_diamond_studies,
    hle_lite_studies,
    mmmlu_lite_studies,
    olympiadbench_studies,
    super_gpqa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13830


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aime24_studies_family(seed: int = _SEED + 0):
    """aime24_studies: synthetic correctness bench."""
    return _finite_blob(aime24_studies.bench_aime24_studies(seed))


def bench_gpqa_diamond_studies_family(seed: int = _SEED + 1):
    """gpqa_diamond_studies: synthetic correctness bench."""
    return _finite_blob(gpqa_diamond_studies.bench_gpqa_diamond_studies(seed))


def bench_hle_lite_studies_family(seed: int = _SEED + 2):
    """hle_lite_studies: synthetic correctness bench."""
    return _finite_blob(hle_lite_studies.bench_hle_lite_studies(seed))


def bench_mmmlu_lite_studies_family(seed: int = _SEED + 3):
    """mmmlu_lite_studies: synthetic correctness bench."""
    return _finite_blob(mmmlu_lite_studies.bench_mmmlu_lite_studies(seed))


def bench_olympiadbench_studies_family(seed: int = _SEED + 4):
    """olympiadbench_studies: synthetic correctness bench."""
    return _finite_blob(olympiadbench_studies.bench_olympiadbench_studies(seed))


def bench_super_gpqa_studies_family(seed: int = _SEED + 5):
    """super_gpqa_studies: synthetic correctness bench."""
    return _finite_blob(super_gpqa_studies.bench_super_gpqa_studies(seed))
