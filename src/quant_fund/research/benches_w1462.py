"""Wave-1462 bench adapters: ocean-life canon (SYNTHETIC only)."""

from quant_fund.models import (
    crab_qa_studies,
    jellyfish_qa_studies,
    octopus_qa_studies,
    seahorse_qa_studies,
    squid_qa_studies,
    stingray_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14620


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_crab_qa_studies_family(seed: int = _SEED + 0):
    """crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(crab_qa_studies.bench_crab_qa_studies(seed))


def bench_jellyfish_qa_studies_family(seed: int = _SEED + 1):
    """jellyfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(jellyfish_qa_studies.bench_jellyfish_qa_studies(seed))


def bench_octopus_qa_studies_family(seed: int = _SEED + 2):
    """octopus_qa_studies: synthetic correctness bench."""
    return _finite_blob(octopus_qa_studies.bench_octopus_qa_studies(seed))


def bench_seahorse_qa_studies_family(seed: int = _SEED + 3):
    """seahorse_qa_studies: synthetic correctness bench."""
    return _finite_blob(seahorse_qa_studies.bench_seahorse_qa_studies(seed))


def bench_squid_qa_studies_family(seed: int = _SEED + 4):
    """squid_qa_studies: synthetic correctness bench."""
    return _finite_blob(squid_qa_studies.bench_squid_qa_studies(seed))


def bench_stingray_qa_studies_family(seed: int = _SEED + 5):
    """stingray_qa_studies: synthetic correctness bench."""
    return _finite_blob(stingray_qa_studies.bench_stingray_qa_studies(seed))
