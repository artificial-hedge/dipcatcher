"""Wave-1948 bench adapters: goetic-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    asmodeus_qa_studies,
    astaroth_qa_studies,
    belial_qa_studies,
    furfur_qa_studies,
    paimon_qa_studies,
    stolas_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19480


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_asmodeus_qa_studies_family(seed: int = _SEED + 0):
    """asmodeus_qa_studies: synthetic correctness bench."""
    return _finite_blob(asmodeus_qa_studies.bench_asmodeus_qa_studies(seed))


def bench_astaroth_qa_studies_family(seed: int = _SEED + 1):
    """astaroth_qa_studies: synthetic correctness bench."""
    return _finite_blob(astaroth_qa_studies.bench_astaroth_qa_studies(seed))


def bench_belial_qa_studies_family(seed: int = _SEED + 2):
    """belial_qa_studies: synthetic correctness bench."""
    return _finite_blob(belial_qa_studies.bench_belial_qa_studies(seed))


def bench_furfur_qa_studies_family(seed: int = _SEED + 3):
    """furfur_qa_studies: synthetic correctness bench."""
    return _finite_blob(furfur_qa_studies.bench_furfur_qa_studies(seed))


def bench_paimon_qa_studies_family(seed: int = _SEED + 4):
    """paimon_qa_studies: synthetic correctness bench."""
    return _finite_blob(paimon_qa_studies.bench_paimon_qa_studies(seed))


def bench_stolas_qa_studies_family(seed: int = _SEED + 5):
    """stolas_qa_studies: synthetic correctness bench."""
    return _finite_blob(stolas_qa_studies.bench_stolas_qa_studies(seed))
