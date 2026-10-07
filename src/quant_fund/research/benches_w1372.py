"""Wave-1372 bench adapters: long-doc-summarization canon (SYNTHETIC only)."""

from quant_fund.models import (
    billsum_lite_studies,
    booksum_lite_studies,
    elm_lite_studies,
    govreport_lite_studies,
    qmsum_lite_studies,
    wikisum_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13720


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_billsum_lite_studies_family(seed: int = _SEED + 0):
    """billsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(billsum_lite_studies.bench_billsum_lite_studies(seed))


def bench_booksum_lite_studies_family(seed: int = _SEED + 1):
    """booksum_lite_studies: synthetic correctness bench."""
    return _finite_blob(booksum_lite_studies.bench_booksum_lite_studies(seed))


def bench_elm_lite_studies_family(seed: int = _SEED + 2):
    """elm_lite_studies: synthetic correctness bench."""
    return _finite_blob(elm_lite_studies.bench_elm_lite_studies(seed))


def bench_govreport_lite_studies_family(seed: int = _SEED + 3):
    """govreport_lite_studies: synthetic correctness bench."""
    return _finite_blob(govreport_lite_studies.bench_govreport_lite_studies(seed))


def bench_qmsum_lite_studies_family(seed: int = _SEED + 4):
    """qmsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(qmsum_lite_studies.bench_qmsum_lite_studies(seed))


def bench_wikisum_lite_studies_family(seed: int = _SEED + 5):
    """wikisum_lite_studies: synthetic correctness bench."""
    return _finite_blob(wikisum_lite_studies.bench_wikisum_lite_studies(seed))
