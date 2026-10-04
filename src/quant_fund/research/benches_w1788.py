"""Wave-1788 bench adapters: roman-rural canon (SYNTHETIC only)."""

from quant_fund.models import (
    ceres_qa_studies,
    flora_qa_studies,
    janus_qa_studies,
    pomona_qa_studies,
    silvanus_qa_studies,
    solinvictus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17880


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ceres_qa_studies_family(seed: int = _SEED + 0):
    """ceres_qa_studies: synthetic correctness bench."""
    return _finite_blob(ceres_qa_studies.bench_ceres_qa_studies(seed))


def bench_flora_qa_studies_family(seed: int = _SEED + 1):
    """flora_qa_studies: synthetic correctness bench."""
    return _finite_blob(flora_qa_studies.bench_flora_qa_studies(seed))


def bench_janus_qa_studies_family(seed: int = _SEED + 2):
    """janus_qa_studies: synthetic correctness bench."""
    return _finite_blob(janus_qa_studies.bench_janus_qa_studies(seed))


def bench_pomona_qa_studies_family(seed: int = _SEED + 3):
    """pomona_qa_studies: synthetic correctness bench."""
    return _finite_blob(pomona_qa_studies.bench_pomona_qa_studies(seed))


def bench_silvanus_qa_studies_family(seed: int = _SEED + 4):
    """silvanus_qa_studies: synthetic correctness bench."""
    return _finite_blob(silvanus_qa_studies.bench_silvanus_qa_studies(seed))


def bench_solinvictus_qa_studies_family(seed: int = _SEED + 5):
    """solinvictus_qa_studies: synthetic correctness bench."""
    return _finite_blob(solinvictus_qa_studies.bench_solinvictus_qa_studies(seed))
