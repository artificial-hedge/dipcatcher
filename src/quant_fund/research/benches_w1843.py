"""Wave-1843 bench adapters: philistine-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    ashtoreth_qa_studies,
    baalzebub_qa_studies,
    dagon2_qa_studies,
    delilah_qa_studies,
    goliath_qa_studies,
    samson_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18430


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ashtoreth_qa_studies_family(seed: int = _SEED + 0):
    """ashtoreth_qa_studies: synthetic correctness bench."""
    return _finite_blob(ashtoreth_qa_studies.bench_ashtoreth_qa_studies(seed))


def bench_baalzebub_qa_studies_family(seed: int = _SEED + 1):
    """baalzebub_qa_studies: synthetic correctness bench."""
    return _finite_blob(baalzebub_qa_studies.bench_baalzebub_qa_studies(seed))


def bench_dagon2_qa_studies_family(seed: int = _SEED + 2):
    """dagon2_qa_studies: synthetic correctness bench."""
    return _finite_blob(dagon2_qa_studies.bench_dagon2_qa_studies(seed))


def bench_delilah_qa_studies_family(seed: int = _SEED + 3):
    """delilah_qa_studies: synthetic correctness bench."""
    return _finite_blob(delilah_qa_studies.bench_delilah_qa_studies(seed))


def bench_goliath_qa_studies_family(seed: int = _SEED + 4):
    """goliath_qa_studies: synthetic correctness bench."""
    return _finite_blob(goliath_qa_studies.bench_goliath_qa_studies(seed))


def bench_samson_qa_studies_family(seed: int = _SEED + 5):
    """samson_qa_studies: synthetic correctness bench."""
    return _finite_blob(samson_qa_studies.bench_samson_qa_studies(seed))
