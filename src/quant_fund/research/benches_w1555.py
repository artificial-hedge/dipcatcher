"""Wave-1555 bench adapters: arachnid-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    harvestman_qa_studies,
    pseudoscorpion_qa_studies,
    solifuge_qa_studies,
    tick_qa_studies,
    vinegaroon_qa_studies,
    whip_scorpion_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15550


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_harvestman_qa_studies_family(seed: int = _SEED + 0):
    """harvestman_qa_studies: synthetic correctness bench."""
    return _finite_blob(harvestman_qa_studies.bench_harvestman_qa_studies(seed))


def bench_pseudoscorpion_qa_studies_family(seed: int = _SEED + 1):
    """pseudoscorpion_qa_studies: synthetic correctness bench."""
    return _finite_blob(pseudoscorpion_qa_studies.bench_pseudoscorpion_qa_studies(seed))


def bench_solifuge_qa_studies_family(seed: int = _SEED + 2):
    """solifuge_qa_studies: synthetic correctness bench."""
    return _finite_blob(solifuge_qa_studies.bench_solifuge_qa_studies(seed))


def bench_tick_qa_studies_family(seed: int = _SEED + 3):
    """tick_qa_studies: synthetic correctness bench."""
    return _finite_blob(tick_qa_studies.bench_tick_qa_studies(seed))


def bench_vinegaroon_qa_studies_family(seed: int = _SEED + 4):
    """vinegaroon_qa_studies: synthetic correctness bench."""
    return _finite_blob(vinegaroon_qa_studies.bench_vinegaroon_qa_studies(seed))


def bench_whip_scorpion_qa_studies_family(seed: int = _SEED + 5):
    """whip_scorpion_qa_studies: synthetic correctness bench."""
    return _finite_blob(whip_scorpion_qa_studies.bench_whip_scorpion_qa_studies(seed))
