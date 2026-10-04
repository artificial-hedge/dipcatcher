"""Wave-1526 bench adapters: fern canon (SYNTHETIC only)."""

from quant_fund.models import (
    bracken_qa_studies,
    horsetail_qa_studies,
    maidenhair_qa_studies,
    staghorn_qa_studies,
    swordfern_qa_studies,
    treefern_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15260


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bracken_qa_studies_family(seed: int = _SEED + 0):
    """bracken_qa_studies: synthetic correctness bench."""
    return _finite_blob(bracken_qa_studies.bench_bracken_qa_studies(seed))


def bench_horsetail_qa_studies_family(seed: int = _SEED + 1):
    """horsetail_qa_studies: synthetic correctness bench."""
    return _finite_blob(horsetail_qa_studies.bench_horsetail_qa_studies(seed))


def bench_maidenhair_qa_studies_family(seed: int = _SEED + 2):
    """maidenhair_qa_studies: synthetic correctness bench."""
    return _finite_blob(maidenhair_qa_studies.bench_maidenhair_qa_studies(seed))


def bench_staghorn_qa_studies_family(seed: int = _SEED + 3):
    """staghorn_qa_studies: synthetic correctness bench."""
    return _finite_blob(staghorn_qa_studies.bench_staghorn_qa_studies(seed))


def bench_swordfern_qa_studies_family(seed: int = _SEED + 4):
    """swordfern_qa_studies: synthetic correctness bench."""
    return _finite_blob(swordfern_qa_studies.bench_swordfern_qa_studies(seed))


def bench_treefern_qa_studies_family(seed: int = _SEED + 5):
    """treefern_qa_studies: synthetic correctness bench."""
    return _finite_blob(treefern_qa_studies.bench_treefern_qa_studies(seed))
