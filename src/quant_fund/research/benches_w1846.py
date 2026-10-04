"""Wave-1846 bench adapters: sabaean-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    almaqah_qa_studies,
    anbay_qa_studies,
    aranyada_qa_studies,
    athtar_qa_studies,
    haubas_qa_studies,
    nasr2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18460


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_almaqah_qa_studies_family(seed: int = _SEED + 0):
    """almaqah_qa_studies: synthetic correctness bench."""
    return _finite_blob(almaqah_qa_studies.bench_almaqah_qa_studies(seed))


def bench_anbay_qa_studies_family(seed: int = _SEED + 1):
    """anbay_qa_studies: synthetic correctness bench."""
    return _finite_blob(anbay_qa_studies.bench_anbay_qa_studies(seed))


def bench_aranyada_qa_studies_family(seed: int = _SEED + 2):
    """aranyada_qa_studies: synthetic correctness bench."""
    return _finite_blob(aranyada_qa_studies.bench_aranyada_qa_studies(seed))


def bench_athtar_qa_studies_family(seed: int = _SEED + 3):
    """athtar_qa_studies: synthetic correctness bench."""
    return _finite_blob(athtar_qa_studies.bench_athtar_qa_studies(seed))


def bench_haubas_qa_studies_family(seed: int = _SEED + 4):
    """haubas_qa_studies: synthetic correctness bench."""
    return _finite_blob(haubas_qa_studies.bench_haubas_qa_studies(seed))


def bench_nasr2_qa_studies_family(seed: int = _SEED + 5):
    """nasr2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nasr2_qa_studies.bench_nasr2_qa_studies(seed))
