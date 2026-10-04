"""Wave-1932 bench adapters: mesoamerican-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    ah_puch_qa_studies,
    alux_qa_studies,
    cizin_qa_studies,
    nahualli_qa_studies,
    vucub_qa_studies,
    xtabay_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19320


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ah_puch_qa_studies_family(seed: int = _SEED + 0):
    """ah_puch_qa_studies: synthetic correctness bench."""
    return _finite_blob(ah_puch_qa_studies.bench_ah_puch_qa_studies(seed))


def bench_alux_qa_studies_family(seed: int = _SEED + 1):
    """alux_qa_studies: synthetic correctness bench."""
    return _finite_blob(alux_qa_studies.bench_alux_qa_studies(seed))


def bench_cizin_qa_studies_family(seed: int = _SEED + 2):
    """cizin_qa_studies: synthetic correctness bench."""
    return _finite_blob(cizin_qa_studies.bench_cizin_qa_studies(seed))


def bench_nahualli_qa_studies_family(seed: int = _SEED + 3):
    """nahualli_qa_studies: synthetic correctness bench."""
    return _finite_blob(nahualli_qa_studies.bench_nahualli_qa_studies(seed))


def bench_vucub_qa_studies_family(seed: int = _SEED + 4):
    """vucub_qa_studies: synthetic correctness bench."""
    return _finite_blob(vucub_qa_studies.bench_vucub_qa_studies(seed))


def bench_xtabay_qa_studies_family(seed: int = _SEED + 5):
    """xtabay_qa_studies: synthetic correctness bench."""
    return _finite_blob(xtabay_qa_studies.bench_xtabay_qa_studies(seed))
