"""Wave-1840 bench adapters: edomite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    atargatis2_qa_studies,
    chemosh2_qa_studies,
    gad2_qa_studies,
    haddad2_qa_studies,
    mot2_qa_studies,
    qos2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18400


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_atargatis2_qa_studies_family(seed: int = _SEED + 0):
    """atargatis2_qa_studies: synthetic correctness bench."""
    return _finite_blob(atargatis2_qa_studies.bench_atargatis2_qa_studies(seed))


def bench_chemosh2_qa_studies_family(seed: int = _SEED + 1):
    """chemosh2_qa_studies: synthetic correctness bench."""
    return _finite_blob(chemosh2_qa_studies.bench_chemosh2_qa_studies(seed))


def bench_gad2_qa_studies_family(seed: int = _SEED + 2):
    """gad2_qa_studies: synthetic correctness bench."""
    return _finite_blob(gad2_qa_studies.bench_gad2_qa_studies(seed))


def bench_haddad2_qa_studies_family(seed: int = _SEED + 3):
    """haddad2_qa_studies: synthetic correctness bench."""
    return _finite_blob(haddad2_qa_studies.bench_haddad2_qa_studies(seed))


def bench_mot2_qa_studies_family(seed: int = _SEED + 4):
    """mot2_qa_studies: synthetic correctness bench."""
    return _finite_blob(mot2_qa_studies.bench_mot2_qa_studies(seed))


def bench_qos2_qa_studies_family(seed: int = _SEED + 5):
    """qos2_qa_studies: synthetic correctness bench."""
    return _finite_blob(qos2_qa_studies.bench_qos2_qa_studies(seed))
