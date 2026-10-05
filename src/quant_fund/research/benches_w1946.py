"""Wave-1946 bench adapters: chinese-underworld canon (SYNTHETIC only)."""

from quant_fund.models import (
    egui_qa_studies,
    heibai_qa_studies,
    meng_po_qa_studies,
    niutou_qa_studies,
    wangliang_qa_studies,
    yanwang_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19460


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_egui_qa_studies_family(seed: int = _SEED + 0):
    """egui_qa_studies: synthetic correctness bench."""
    return _finite_blob(egui_qa_studies.bench_egui_qa_studies(seed))


def bench_heibai_qa_studies_family(seed: int = _SEED + 1):
    """heibai_qa_studies: synthetic correctness bench."""
    return _finite_blob(heibai_qa_studies.bench_heibai_qa_studies(seed))


def bench_meng_po_qa_studies_family(seed: int = _SEED + 2):
    """meng_po_qa_studies: synthetic correctness bench."""
    return _finite_blob(meng_po_qa_studies.bench_meng_po_qa_studies(seed))


def bench_niutou_qa_studies_family(seed: int = _SEED + 3):
    """niutou_qa_studies: synthetic correctness bench."""
    return _finite_blob(niutou_qa_studies.bench_niutou_qa_studies(seed))


def bench_wangliang_qa_studies_family(seed: int = _SEED + 4):
    """wangliang_qa_studies: synthetic correctness bench."""
    return _finite_blob(wangliang_qa_studies.bench_wangliang_qa_studies(seed))


def bench_yanwang_qa_studies_family(seed: int = _SEED + 5):
    """yanwang_qa_studies: synthetic correctness bench."""
    return _finite_blob(yanwang_qa_studies.bench_yanwang_qa_studies(seed))
