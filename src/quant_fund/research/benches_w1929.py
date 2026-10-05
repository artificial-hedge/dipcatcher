"""Wave-1929 bench adapters: maori-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    hotupuku_qa_studies,
    kahui_tipua_qa_studies,
    kataore_qa_studies,
    nuku_mai_tore_qa_studies,
    tipua_qa_studies,
    wheke_muturangi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19290


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hotupuku_qa_studies_family(seed: int = _SEED + 0):
    """hotupuku_qa_studies: synthetic correctness bench."""
    return _finite_blob(hotupuku_qa_studies.bench_hotupuku_qa_studies(seed))


def bench_kahui_tipua_qa_studies_family(seed: int = _SEED + 1):
    """kahui_tipua_qa_studies: synthetic correctness bench."""
    return _finite_blob(kahui_tipua_qa_studies.bench_kahui_tipua_qa_studies(seed))


def bench_kataore_qa_studies_family(seed: int = _SEED + 2):
    """kataore_qa_studies: synthetic correctness bench."""
    return _finite_blob(kataore_qa_studies.bench_kataore_qa_studies(seed))


def bench_nuku_mai_tore_qa_studies_family(seed: int = _SEED + 3):
    """nuku_mai_tore_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuku_mai_tore_qa_studies.bench_nuku_mai_tore_qa_studies(seed))


def bench_tipua_qa_studies_family(seed: int = _SEED + 4):
    """tipua_qa_studies: synthetic correctness bench."""
    return _finite_blob(tipua_qa_studies.bench_tipua_qa_studies(seed))


def bench_wheke_muturangi_qa_studies_family(seed: int = _SEED + 5):
    """wheke_muturangi_qa_studies: synthetic correctness bench."""
    return _finite_blob(wheke_muturangi_qa_studies.bench_wheke_muturangi_qa_studies(seed))
