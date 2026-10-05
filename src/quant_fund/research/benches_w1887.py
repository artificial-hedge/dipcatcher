"""Wave-1887 bench adapters: indo-iranian canon (SYNTHETIC only)."""

from quant_fund.models import (
    dakini_qa_studies,
    indra_hindu_qa_studies,
    jamshid_qa_studies,
    varuna_qa_studies,
    vritra_qa_studies,
    zurvan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18870


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dakini_qa_studies_family(seed: int = _SEED + 0):
    """dakini_qa_studies: synthetic correctness bench."""
    return _finite_blob(dakini_qa_studies.bench_dakini_qa_studies(seed))


def bench_indra_hindu_qa_studies_family(seed: int = _SEED + 1):
    """indra_hindu_qa_studies: synthetic correctness bench."""
    return _finite_blob(indra_hindu_qa_studies.bench_indra_hindu_qa_studies(seed))


def bench_jamshid_qa_studies_family(seed: int = _SEED + 2):
    """jamshid_qa_studies: synthetic correctness bench."""
    return _finite_blob(jamshid_qa_studies.bench_jamshid_qa_studies(seed))


def bench_varuna_qa_studies_family(seed: int = _SEED + 3):
    """varuna_qa_studies: synthetic correctness bench."""
    return _finite_blob(varuna_qa_studies.bench_varuna_qa_studies(seed))


def bench_vritra_qa_studies_family(seed: int = _SEED + 4):
    """vritra_qa_studies: synthetic correctness bench."""
    return _finite_blob(vritra_qa_studies.bench_vritra_qa_studies(seed))


def bench_zurvan_qa_studies_family(seed: int = _SEED + 5):
    """zurvan_qa_studies: synthetic correctness bench."""
    return _finite_blob(zurvan_qa_studies.bench_zurvan_qa_studies(seed))
