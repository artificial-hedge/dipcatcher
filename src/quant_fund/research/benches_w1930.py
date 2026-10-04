"""Wave-1930 bench adapters: tibetan-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    bdud_qa_studies,
    bgegs_qa_studies,
    btsan_qa_studies,
    gdon_qa_studies,
    gnod_sbyin_qa_studies,
    srin_po_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19300


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bdud_qa_studies_family(seed: int = _SEED + 0):
    """bdud_qa_studies: synthetic correctness bench."""
    return _finite_blob(bdud_qa_studies.bench_bdud_qa_studies(seed))


def bench_bgegs_qa_studies_family(seed: int = _SEED + 1):
    """bgegs_qa_studies: synthetic correctness bench."""
    return _finite_blob(bgegs_qa_studies.bench_bgegs_qa_studies(seed))


def bench_btsan_qa_studies_family(seed: int = _SEED + 2):
    """btsan_qa_studies: synthetic correctness bench."""
    return _finite_blob(btsan_qa_studies.bench_btsan_qa_studies(seed))


def bench_gdon_qa_studies_family(seed: int = _SEED + 3):
    """gdon_qa_studies: synthetic correctness bench."""
    return _finite_blob(gdon_qa_studies.bench_gdon_qa_studies(seed))


def bench_gnod_sbyin_qa_studies_family(seed: int = _SEED + 4):
    """gnod_sbyin_qa_studies: synthetic correctness bench."""
    return _finite_blob(gnod_sbyin_qa_studies.bench_gnod_sbyin_qa_studies(seed))


def bench_srin_po_qa_studies_family(seed: int = _SEED + 5):
    """srin_po_qa_studies: synthetic correctness bench."""
    return _finite_blob(srin_po_qa_studies.bench_srin_po_qa_studies(seed))
