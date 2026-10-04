"""Wave-1786 bench adapters: mesopotamian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ea_qa_studies,
    humbaba_qa_studies,
    pazuzu_qa_studies,
    sargon_qa_studies,
    semiramis_qa_studies,
    utnapishtim_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17860


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ea_qa_studies_family(seed: int = _SEED + 0):
    """ea_qa_studies: synthetic correctness bench."""
    return _finite_blob(ea_qa_studies.bench_ea_qa_studies(seed))


def bench_humbaba_qa_studies_family(seed: int = _SEED + 1):
    """humbaba_qa_studies: synthetic correctness bench."""
    return _finite_blob(humbaba_qa_studies.bench_humbaba_qa_studies(seed))


def bench_pazuzu_qa_studies_family(seed: int = _SEED + 2):
    """pazuzu_qa_studies: synthetic correctness bench."""
    return _finite_blob(pazuzu_qa_studies.bench_pazuzu_qa_studies(seed))


def bench_sargon_qa_studies_family(seed: int = _SEED + 3):
    """sargon_qa_studies: synthetic correctness bench."""
    return _finite_blob(sargon_qa_studies.bench_sargon_qa_studies(seed))


def bench_semiramis_qa_studies_family(seed: int = _SEED + 4):
    """semiramis_qa_studies: synthetic correctness bench."""
    return _finite_blob(semiramis_qa_studies.bench_semiramis_qa_studies(seed))


def bench_utnapishtim_qa_studies_family(seed: int = _SEED + 5):
    """utnapishtim_qa_studies: synthetic correctness bench."""
    return _finite_blob(utnapishtim_qa_studies.bench_utnapishtim_qa_studies(seed))
