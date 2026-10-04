"""Wave-1596 bench adapters: ocean-mammal canon (SYNTHETIC only)."""

from quant_fund.models import (
    bottlenose_qa_studies,
    dusky_dolphin_qa_studies,
    false_killer_qa_studies,
    melon_head_qa_studies,
    pygmy_whale_qa_studies,
    sea_lion_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15960


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bottlenose_qa_studies_family(seed: int = _SEED + 0):
    """bottlenose_qa_studies: synthetic correctness bench."""
    return _finite_blob(bottlenose_qa_studies.bench_bottlenose_qa_studies(seed))


def bench_dusky_dolphin_qa_studies_family(seed: int = _SEED + 1):
    """dusky_dolphin_qa_studies: synthetic correctness bench."""
    return _finite_blob(dusky_dolphin_qa_studies.bench_dusky_dolphin_qa_studies(seed))


def bench_false_killer_qa_studies_family(seed: int = _SEED + 2):
    """false_killer_qa_studies: synthetic correctness bench."""
    return _finite_blob(false_killer_qa_studies.bench_false_killer_qa_studies(seed))


def bench_melon_head_qa_studies_family(seed: int = _SEED + 3):
    """melon_head_qa_studies: synthetic correctness bench."""
    return _finite_blob(melon_head_qa_studies.bench_melon_head_qa_studies(seed))


def bench_pygmy_whale_qa_studies_family(seed: int = _SEED + 4):
    """pygmy_whale_qa_studies: synthetic correctness bench."""
    return _finite_blob(pygmy_whale_qa_studies.bench_pygmy_whale_qa_studies(seed))


def bench_sea_lion_qa_studies_family(seed: int = _SEED + 5):
    """sea_lion_qa_studies: synthetic correctness bench."""
    return _finite_blob(sea_lion_qa_studies.bench_sea_lion_qa_studies(seed))
