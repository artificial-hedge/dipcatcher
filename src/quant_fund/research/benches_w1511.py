"""Wave-1511 bench adapters: hummingbird canon (SYNTHETIC only)."""

from quant_fund.models import (
    brilliant_qa_studies,
    hermit_qa_studies,
    hummingbird_qa_studies,
    sapphire_qa_studies,
    topaz_qa_studies,
    woodstar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15110


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_brilliant_qa_studies_family(seed: int = _SEED + 0):
    """brilliant_qa_studies: synthetic correctness bench."""
    return _finite_blob(brilliant_qa_studies.bench_brilliant_qa_studies(seed))


def bench_hermit_qa_studies_family(seed: int = _SEED + 1):
    """hermit_qa_studies: synthetic correctness bench."""
    return _finite_blob(hermit_qa_studies.bench_hermit_qa_studies(seed))


def bench_hummingbird_qa_studies_family(seed: int = _SEED + 2):
    """hummingbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(hummingbird_qa_studies.bench_hummingbird_qa_studies(seed))


def bench_sapphire_qa_studies_family(seed: int = _SEED + 3):
    """sapphire_qa_studies: synthetic correctness bench."""
    return _finite_blob(sapphire_qa_studies.bench_sapphire_qa_studies(seed))


def bench_topaz_qa_studies_family(seed: int = _SEED + 4):
    """topaz_qa_studies: synthetic correctness bench."""
    return _finite_blob(topaz_qa_studies.bench_topaz_qa_studies(seed))


def bench_woodstar_qa_studies_family(seed: int = _SEED + 5):
    """woodstar_qa_studies: synthetic correctness bench."""
    return _finite_blob(woodstar_qa_studies.bench_woodstar_qa_studies(seed))
