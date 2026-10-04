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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
