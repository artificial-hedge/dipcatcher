"""Wave-1493 bench adapters: antelope-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bongo_qa_studies,
    duiker_qa_studies,
    hartebeest_qa_studies,
    nyala_qa_studies,
    topi_qa_studies,
    waterbuck_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14930


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bongo_qa_studies_family(seed: int = _SEED + 0):
    """bongo_qa_studies: synthetic correctness bench."""
    return _finite_blob(bongo_qa_studies.bench_bongo_qa_studies(seed))


def bench_duiker_qa_studies_family(seed: int = _SEED + 1):
    """duiker_qa_studies: synthetic correctness bench."""
    return _finite_blob(duiker_qa_studies.bench_duiker_qa_studies(seed))


def bench_hartebeest_qa_studies_family(seed: int = _SEED + 2):
    """hartebeest_qa_studies: synthetic correctness bench."""
    return _finite_blob(hartebeest_qa_studies.bench_hartebeest_qa_studies(seed))


def bench_nyala_qa_studies_family(seed: int = _SEED + 3):
    """nyala_qa_studies: synthetic correctness bench."""
    return _finite_blob(nyala_qa_studies.bench_nyala_qa_studies(seed))


def bench_topi_qa_studies_family(seed: int = _SEED + 4):
    """topi_qa_studies: synthetic correctness bench."""
    return _finite_blob(topi_qa_studies.bench_topi_qa_studies(seed))


def bench_waterbuck_qa_studies_family(seed: int = _SEED + 5):
    """waterbuck_qa_studies: synthetic correctness bench."""
    return _finite_blob(waterbuck_qa_studies.bench_waterbuck_qa_studies(seed))
