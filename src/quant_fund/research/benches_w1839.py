"""Wave-1839 bench adapters: nabataean-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    alkutba2_qa_studies,
    aluzza2_qa_studies,
    dushara2_qa_studies,
    godil2_qa_studies,
    hubal2_qa_studies,
    manat2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18390


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alkutba2_qa_studies_family(seed: int = _SEED + 0):
    """alkutba2_qa_studies: synthetic correctness bench."""
    return _finite_blob(alkutba2_qa_studies.bench_alkutba2_qa_studies(seed))


def bench_aluzza2_qa_studies_family(seed: int = _SEED + 1):
    """aluzza2_qa_studies: synthetic correctness bench."""
    return _finite_blob(aluzza2_qa_studies.bench_aluzza2_qa_studies(seed))


def bench_dushara2_qa_studies_family(seed: int = _SEED + 2):
    """dushara2_qa_studies: synthetic correctness bench."""
    return _finite_blob(dushara2_qa_studies.bench_dushara2_qa_studies(seed))


def bench_godil2_qa_studies_family(seed: int = _SEED + 3):
    """godil2_qa_studies: synthetic correctness bench."""
    return _finite_blob(godil2_qa_studies.bench_godil2_qa_studies(seed))


def bench_hubal2_qa_studies_family(seed: int = _SEED + 4):
    """hubal2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hubal2_qa_studies.bench_hubal2_qa_studies(seed))


def bench_manat2_qa_studies_family(seed: int = _SEED + 5):
    """manat2_qa_studies: synthetic correctness bench."""
    return _finite_blob(manat2_qa_studies.bench_manat2_qa_studies(seed))
