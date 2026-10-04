"""Wave-1748 bench adapters: tibetan-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    beg_tse_qa_studies,
    dorje_legpa_qa_studies,
    palden_lhamo_qa_studies,
    pehar_qa_studies,
    tsen_god_qa_studies,
    tsiu_marpo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17480


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_beg_tse_qa_studies_family(seed: int = _SEED + 0):
    """beg_tse_qa_studies: synthetic correctness bench."""
    return _finite_blob(beg_tse_qa_studies.bench_beg_tse_qa_studies(seed))


def bench_dorje_legpa_qa_studies_family(seed: int = _SEED + 1):
    """dorje_legpa_qa_studies: synthetic correctness bench."""
    return _finite_blob(dorje_legpa_qa_studies.bench_dorje_legpa_qa_studies(seed))


def bench_palden_lhamo_qa_studies_family(seed: int = _SEED + 2):
    """palden_lhamo_qa_studies: synthetic correctness bench."""
    return _finite_blob(palden_lhamo_qa_studies.bench_palden_lhamo_qa_studies(seed))


def bench_pehar_qa_studies_family(seed: int = _SEED + 3):
    """pehar_qa_studies: synthetic correctness bench."""
    return _finite_blob(pehar_qa_studies.bench_pehar_qa_studies(seed))


def bench_tsen_god_qa_studies_family(seed: int = _SEED + 4):
    """tsen_god_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsen_god_qa_studies.bench_tsen_god_qa_studies(seed))


def bench_tsiu_marpo_qa_studies_family(seed: int = _SEED + 5):
    """tsiu_marpo_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsiu_marpo_qa_studies.bench_tsiu_marpo_qa_studies(seed))
