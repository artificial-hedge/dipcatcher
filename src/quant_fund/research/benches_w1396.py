"""Wave-1396 bench adapters: summarization-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    agnews_lite_studies,
    dialsum_lite_studies,
    facet_lite_studies,
    medsum_lite_studies,
    oposum_lite_studies,
    qsum_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13960


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


def bench_agnews_lite_studies_family(seed: int = _SEED + 0):
    """agnews_lite_studies: synthetic correctness bench."""
    return _finite_blob(agnews_lite_studies.bench_agnews_lite_studies(seed))


def bench_dialsum_lite_studies_family(seed: int = _SEED + 1):
    """dialsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(dialsum_lite_studies.bench_dialsum_lite_studies(seed))


def bench_facet_lite_studies_family(seed: int = _SEED + 2):
    """facet_lite_studies: synthetic correctness bench."""
    return _finite_blob(facet_lite_studies.bench_facet_lite_studies(seed))


def bench_medsum_lite_studies_family(seed: int = _SEED + 3):
    """medsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(medsum_lite_studies.bench_medsum_lite_studies(seed))


def bench_oposum_lite_studies_family(seed: int = _SEED + 4):
    """oposum_lite_studies: synthetic correctness bench."""
    return _finite_blob(oposum_lite_studies.bench_oposum_lite_studies(seed))


def bench_qsum_lite_studies_family(seed: int = _SEED + 5):
    """qsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(qsum_lite_studies.bench_qsum_lite_studies(seed))
