"""Wave-1856 bench adapters: gallic-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    cernunnos_qa_studies,
    epona_qa_studies,
    esus_qa_studies,
    rosmerta_qa_studies,
    taranis_qa_studies,
    teutates_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18560


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cernunnos_qa_studies_family(seed: int = _SEED + 0):
    """cernunnos_qa_studies: synthetic correctness bench."""
    return _finite_blob(cernunnos_qa_studies.bench_cernunnos_qa_studies(seed))


def bench_epona_qa_studies_family(seed: int = _SEED + 1):
    """epona_qa_studies: synthetic correctness bench."""
    return _finite_blob(epona_qa_studies.bench_epona_qa_studies(seed))


def bench_esus_qa_studies_family(seed: int = _SEED + 2):
    """esus_qa_studies: synthetic correctness bench."""
    return _finite_blob(esus_qa_studies.bench_esus_qa_studies(seed))


def bench_rosmerta_qa_studies_family(seed: int = _SEED + 3):
    """rosmerta_qa_studies: synthetic correctness bench."""
    return _finite_blob(rosmerta_qa_studies.bench_rosmerta_qa_studies(seed))


def bench_taranis_qa_studies_family(seed: int = _SEED + 4):
    """taranis_qa_studies: synthetic correctness bench."""
    return _finite_blob(taranis_qa_studies.bench_taranis_qa_studies(seed))


def bench_teutates_qa_studies_family(seed: int = _SEED + 5):
    """teutates_qa_studies: synthetic correctness bench."""
    return _finite_blob(teutates_qa_studies.bench_teutates_qa_studies(seed))
