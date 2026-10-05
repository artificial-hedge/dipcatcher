"""Wave-1910 bench adapters: persian-div canon (SYNTHETIC only)."""

from quant_fund.models import (
    demon_div_qa_studies,
    div_aq_qa_studies,
    div_demon_qa_studies,
    druj_spirit_qa_studies,
    nasu_demon_qa_studies,
    yalburz_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19100


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_demon_div_qa_studies_family(seed: int = _SEED + 0):
    """demon_div_qa_studies: synthetic correctness bench."""
    return _finite_blob(demon_div_qa_studies.bench_demon_div_qa_studies(seed))


def bench_div_aq_qa_studies_family(seed: int = _SEED + 1):
    """div_aq_qa_studies: synthetic correctness bench."""
    return _finite_blob(div_aq_qa_studies.bench_div_aq_qa_studies(seed))


def bench_div_demon_qa_studies_family(seed: int = _SEED + 2):
    """div_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(div_demon_qa_studies.bench_div_demon_qa_studies(seed))


def bench_druj_spirit_qa_studies_family(seed: int = _SEED + 3):
    """druj_spirit_qa_studies: synthetic correctness bench."""
    return _finite_blob(druj_spirit_qa_studies.bench_druj_spirit_qa_studies(seed))


def bench_nasu_demon_qa_studies_family(seed: int = _SEED + 4):
    """nasu_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(nasu_demon_qa_studies.bench_nasu_demon_qa_studies(seed))


def bench_yalburz_qa_studies_family(seed: int = _SEED + 5):
    """yalburz_qa_studies: synthetic correctness bench."""
    return _finite_blob(yalburz_qa_studies.bench_yalburz_qa_studies(seed))
