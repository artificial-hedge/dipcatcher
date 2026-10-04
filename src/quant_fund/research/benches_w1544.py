"""Wave-1544 bench adapters: coraciiform canon (SYNTHETIC only)."""

from quant_fund.models import (
    hoopoe_qa_studies,
    nunbird_qa_studies,
    nunlet_qa_studies,
    puffbird_qa_studies,
    toco_qa_studies,
    woodhoopoe_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15440


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hoopoe_qa_studies_family(seed: int = _SEED + 0):
    """hoopoe_qa_studies: synthetic correctness bench."""
    return _finite_blob(hoopoe_qa_studies.bench_hoopoe_qa_studies(seed))


def bench_nunbird_qa_studies_family(seed: int = _SEED + 1):
    """nunbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(nunbird_qa_studies.bench_nunbird_qa_studies(seed))


def bench_nunlet_qa_studies_family(seed: int = _SEED + 2):
    """nunlet_qa_studies: synthetic correctness bench."""
    return _finite_blob(nunlet_qa_studies.bench_nunlet_qa_studies(seed))


def bench_puffbird_qa_studies_family(seed: int = _SEED + 3):
    """puffbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(puffbird_qa_studies.bench_puffbird_qa_studies(seed))


def bench_toco_qa_studies_family(seed: int = _SEED + 4):
    """toco_qa_studies: synthetic correctness bench."""
    return _finite_blob(toco_qa_studies.bench_toco_qa_studies(seed))


def bench_woodhoopoe_qa_studies_family(seed: int = _SEED + 5):
    """woodhoopoe_qa_studies: synthetic correctness bench."""
    return _finite_blob(woodhoopoe_qa_studies.bench_woodhoopoe_qa_studies(seed))
