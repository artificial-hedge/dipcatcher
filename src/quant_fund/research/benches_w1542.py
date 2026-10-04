"""Wave-1542 bench adapters: raptor-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    accipiter_qa_studies,
    bateleur_qa_studies,
    falconet_qa_studies,
    harpy_qa_studies,
    lammergeier_qa_studies,
    seriema_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15420


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_accipiter_qa_studies_family(seed: int = _SEED + 0):
    """accipiter_qa_studies: synthetic correctness bench."""
    return _finite_blob(accipiter_qa_studies.bench_accipiter_qa_studies(seed))


def bench_bateleur_qa_studies_family(seed: int = _SEED + 1):
    """bateleur_qa_studies: synthetic correctness bench."""
    return _finite_blob(bateleur_qa_studies.bench_bateleur_qa_studies(seed))


def bench_falconet_qa_studies_family(seed: int = _SEED + 2):
    """falconet_qa_studies: synthetic correctness bench."""
    return _finite_blob(falconet_qa_studies.bench_falconet_qa_studies(seed))


def bench_harpy_qa_studies_family(seed: int = _SEED + 3):
    """harpy_qa_studies: synthetic correctness bench."""
    return _finite_blob(harpy_qa_studies.bench_harpy_qa_studies(seed))


def bench_lammergeier_qa_studies_family(seed: int = _SEED + 4):
    """lammergeier_qa_studies: synthetic correctness bench."""
    return _finite_blob(lammergeier_qa_studies.bench_lammergeier_qa_studies(seed))


def bench_seriema_qa_studies_family(seed: int = _SEED + 5):
    """seriema_qa_studies: synthetic correctness bench."""
    return _finite_blob(seriema_qa_studies.bench_seriema_qa_studies(seed))
