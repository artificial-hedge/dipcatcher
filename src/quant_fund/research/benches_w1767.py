"""Wave-1767 bench adapters: incan-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    coniraya_qa_studies,
    guanare_qa_studies,
    inti_qa_studies,
    pachacamac_qa_studies,
    supay_qa_studies,
    viracocha_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17670


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_coniraya_qa_studies_family(seed: int = _SEED + 0):
    """coniraya_qa_studies: synthetic correctness bench."""
    return _finite_blob(coniraya_qa_studies.bench_coniraya_qa_studies(seed))


def bench_guanare_qa_studies_family(seed: int = _SEED + 1):
    """guanare_qa_studies: synthetic correctness bench."""
    return _finite_blob(guanare_qa_studies.bench_guanare_qa_studies(seed))


def bench_inti_qa_studies_family(seed: int = _SEED + 2):
    """inti_qa_studies: synthetic correctness bench."""
    return _finite_blob(inti_qa_studies.bench_inti_qa_studies(seed))


def bench_pachacamac_qa_studies_family(seed: int = _SEED + 3):
    """pachacamac_qa_studies: synthetic correctness bench."""
    return _finite_blob(pachacamac_qa_studies.bench_pachacamac_qa_studies(seed))


def bench_supay_qa_studies_family(seed: int = _SEED + 4):
    """supay_qa_studies: synthetic correctness bench."""
    return _finite_blob(supay_qa_studies.bench_supay_qa_studies(seed))


def bench_viracocha_qa_studies_family(seed: int = _SEED + 5):
    """viracocha_qa_studies: synthetic correctness bench."""
    return _finite_blob(viracocha_qa_studies.bench_viracocha_qa_studies(seed))
