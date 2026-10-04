"""Wave-1537 bench adapters: marshbird canon (SYNTHETIC only)."""

from quant_fund.models import (
    coot_qa_studies,
    crake_qa_studies,
    dabchick_qa_studies,
    gallinule_qa_studies,
    rail_qa_studies,
    waterhen_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15370


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_coot_qa_studies_family(seed: int = _SEED + 0):
    """coot_qa_studies: synthetic correctness bench."""
    return _finite_blob(coot_qa_studies.bench_coot_qa_studies(seed))


def bench_crake_qa_studies_family(seed: int = _SEED + 1):
    """crake_qa_studies: synthetic correctness bench."""
    return _finite_blob(crake_qa_studies.bench_crake_qa_studies(seed))


def bench_dabchick_qa_studies_family(seed: int = _SEED + 2):
    """dabchick_qa_studies: synthetic correctness bench."""
    return _finite_blob(dabchick_qa_studies.bench_dabchick_qa_studies(seed))


def bench_gallinule_qa_studies_family(seed: int = _SEED + 3):
    """gallinule_qa_studies: synthetic correctness bench."""
    return _finite_blob(gallinule_qa_studies.bench_gallinule_qa_studies(seed))


def bench_rail_qa_studies_family(seed: int = _SEED + 4):
    """rail_qa_studies: synthetic correctness bench."""
    return _finite_blob(rail_qa_studies.bench_rail_qa_studies(seed))


def bench_waterhen_qa_studies_family(seed: int = _SEED + 5):
    """waterhen_qa_studies: synthetic correctness bench."""
    return _finite_blob(waterhen_qa_studies.bench_waterhen_qa_studies(seed))
