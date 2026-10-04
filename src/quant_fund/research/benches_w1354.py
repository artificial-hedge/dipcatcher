"""Wave-1354 bench adapters: MC-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    arc_easy2_studies,
    boolq_lite_studies,
    cosmos_qa_studies,
    race_lite_studies,
    sciq_lite_studies,
    social_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13540


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arc_easy2_studies_family(seed: int = _SEED + 0):
    """arc_easy2_studies: synthetic correctness bench."""
    return _finite_blob(arc_easy2_studies.bench_arc_easy2_studies(seed))


def bench_boolq_lite_studies_family(seed: int = _SEED + 1):
    """boolq_lite_studies: synthetic correctness bench."""
    return _finite_blob(boolq_lite_studies.bench_boolq_lite_studies(seed))


def bench_cosmos_qa_studies_family(seed: int = _SEED + 2):
    """cosmos_qa_studies: synthetic correctness bench."""
    return _finite_blob(cosmos_qa_studies.bench_cosmos_qa_studies(seed))


def bench_race_lite_studies_family(seed: int = _SEED + 3):
    """race_lite_studies: synthetic correctness bench."""
    return _finite_blob(race_lite_studies.bench_race_lite_studies(seed))


def bench_sciq_lite_studies_family(seed: int = _SEED + 4):
    """sciq_lite_studies: synthetic correctness bench."""
    return _finite_blob(sciq_lite_studies.bench_sciq_lite_studies(seed))


def bench_social_qa_studies_family(seed: int = _SEED + 5):
    """social_qa_studies: synthetic correctness bench."""
    return _finite_blob(social_qa_studies.bench_social_qa_studies(seed))
