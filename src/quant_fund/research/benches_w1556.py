"""Wave-1556 bench adapters: detritivore canon (SYNTHETIC only)."""

from quant_fund.models import (
    bristletail_qa_studies,
    pillbug_qa_studies,
    silverfish_qa_studies,
    springtail_qa_studies,
    velvet_worm_qa_studies,
    woodlouse_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15560


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bristletail_qa_studies_family(seed: int = _SEED + 0):
    """bristletail_qa_studies: synthetic correctness bench."""
    return _finite_blob(bristletail_qa_studies.bench_bristletail_qa_studies(seed))


def bench_pillbug_qa_studies_family(seed: int = _SEED + 1):
    """pillbug_qa_studies: synthetic correctness bench."""
    return _finite_blob(pillbug_qa_studies.bench_pillbug_qa_studies(seed))


def bench_silverfish_qa_studies_family(seed: int = _SEED + 2):
    """silverfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(silverfish_qa_studies.bench_silverfish_qa_studies(seed))


def bench_springtail_qa_studies_family(seed: int = _SEED + 3):
    """springtail_qa_studies: synthetic correctness bench."""
    return _finite_blob(springtail_qa_studies.bench_springtail_qa_studies(seed))


def bench_velvet_worm_qa_studies_family(seed: int = _SEED + 4):
    """velvet_worm_qa_studies: synthetic correctness bench."""
    return _finite_blob(velvet_worm_qa_studies.bench_velvet_worm_qa_studies(seed))


def bench_woodlouse_qa_studies_family(seed: int = _SEED + 5):
    """woodlouse_qa_studies: synthetic correctness bench."""
    return _finite_blob(woodlouse_qa_studies.bench_woodlouse_qa_studies(seed))
