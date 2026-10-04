"""Wave-1690 bench adapters: slavic-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    kladenets_qa_studies,
    kostroma_qa_studies,
    leshii_qa_studies,
    morozko_qa_studies,
    vedmak_qa_studies,
    yarilo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16900


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_kladenets_qa_studies_family(seed: int = _SEED + 0):
    """kladenets_qa_studies: synthetic correctness bench."""
    return _finite_blob(kladenets_qa_studies.bench_kladenets_qa_studies(seed))


def bench_kostroma_qa_studies_family(seed: int = _SEED + 1):
    """kostroma_qa_studies: synthetic correctness bench."""
    return _finite_blob(kostroma_qa_studies.bench_kostroma_qa_studies(seed))


def bench_leshii_qa_studies_family(seed: int = _SEED + 2):
    """leshii_qa_studies: synthetic correctness bench."""
    return _finite_blob(leshii_qa_studies.bench_leshii_qa_studies(seed))


def bench_morozko_qa_studies_family(seed: int = _SEED + 3):
    """morozko_qa_studies: synthetic correctness bench."""
    return _finite_blob(morozko_qa_studies.bench_morozko_qa_studies(seed))


def bench_vedmak_qa_studies_family(seed: int = _SEED + 4):
    """vedmak_qa_studies: synthetic correctness bench."""
    return _finite_blob(vedmak_qa_studies.bench_vedmak_qa_studies(seed))


def bench_yarilo_qa_studies_family(seed: int = _SEED + 5):
    """yarilo_qa_studies: synthetic correctness bench."""
    return _finite_blob(yarilo_qa_studies.bench_yarilo_qa_studies(seed))
