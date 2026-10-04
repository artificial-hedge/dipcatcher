"""Wave-1656 bench adapters: celtic-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    aatxe_qa_studies,
    achiyalabopa_qa_studies,
    afanc_qa_studies,
    akhlut_qa_studies,
    amarok_qa_studies,
    eachuisge_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16560


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aatxe_qa_studies_family(seed: int = _SEED + 0):
    """aatxe_qa_studies: synthetic correctness bench."""
    return _finite_blob(aatxe_qa_studies.bench_aatxe_qa_studies(seed))


def bench_achiyalabopa_qa_studies_family(seed: int = _SEED + 1):
    """achiyalabopa_qa_studies: synthetic correctness bench."""
    return _finite_blob(achiyalabopa_qa_studies.bench_achiyalabopa_qa_studies(seed))


def bench_afanc_qa_studies_family(seed: int = _SEED + 2):
    """afanc_qa_studies: synthetic correctness bench."""
    return _finite_blob(afanc_qa_studies.bench_afanc_qa_studies(seed))


def bench_akhlut_qa_studies_family(seed: int = _SEED + 3):
    """akhlut_qa_studies: synthetic correctness bench."""
    return _finite_blob(akhlut_qa_studies.bench_akhlut_qa_studies(seed))


def bench_amarok_qa_studies_family(seed: int = _SEED + 4):
    """amarok_qa_studies: synthetic correctness bench."""
    return _finite_blob(amarok_qa_studies.bench_amarok_qa_studies(seed))


def bench_eachuisge_qa_studies_family(seed: int = _SEED + 5):
    """eachuisge_qa_studies: synthetic correctness bench."""
    return _finite_blob(eachuisge_qa_studies.bench_eachuisge_qa_studies(seed))
