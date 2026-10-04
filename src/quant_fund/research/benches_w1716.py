"""Wave-1716 bench adapters: thracian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    heroas_qa_studies,
    kottiso_qa_studies,
    kotys_qa_studies,
    semele_qa_studies,
    theandrites_qa_studies,
    zibelthiurdos_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17160


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_heroas_qa_studies_family(seed: int = _SEED + 0):
    """heroas_qa_studies: synthetic correctness bench."""
    return _finite_blob(heroas_qa_studies.bench_heroas_qa_studies(seed))


def bench_kottiso_qa_studies_family(seed: int = _SEED + 1):
    """kottiso_qa_studies: synthetic correctness bench."""
    return _finite_blob(kottiso_qa_studies.bench_kottiso_qa_studies(seed))


def bench_kotys_qa_studies_family(seed: int = _SEED + 2):
    """kotys_qa_studies: synthetic correctness bench."""
    return _finite_blob(kotys_qa_studies.bench_kotys_qa_studies(seed))


def bench_semele_qa_studies_family(seed: int = _SEED + 3):
    """semele_qa_studies: synthetic correctness bench."""
    return _finite_blob(semele_qa_studies.bench_semele_qa_studies(seed))


def bench_theandrites_qa_studies_family(seed: int = _SEED + 4):
    """theandrites_qa_studies: synthetic correctness bench."""
    return _finite_blob(theandrites_qa_studies.bench_theandrites_qa_studies(seed))


def bench_zibelthiurdos_qa_studies_family(seed: int = _SEED + 5):
    """zibelthiurdos_qa_studies: synthetic correctness bench."""
    return _finite_blob(zibelthiurdos_qa_studies.bench_zibelthiurdos_qa_studies(seed))
