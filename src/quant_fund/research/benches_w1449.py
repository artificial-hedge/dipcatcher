"""Wave-1449 bench adapters: marine mammal canon (SYNTHETIC only)."""

from quant_fund.models import (
    beluga_qa_studies,
    manatee_qa_studies,
    narwhal_qa_studies,
    orca_qa_studies,
    otter_qa_studies,
    walrus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14490


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_beluga_qa_studies_family(seed: int = _SEED + 0):
    """beluga_qa_studies: synthetic correctness bench."""
    return _finite_blob(beluga_qa_studies.bench_beluga_qa_studies(seed))


def bench_manatee_qa_studies_family(seed: int = _SEED + 1):
    """manatee_qa_studies: synthetic correctness bench."""
    return _finite_blob(manatee_qa_studies.bench_manatee_qa_studies(seed))


def bench_narwhal_qa_studies_family(seed: int = _SEED + 2):
    """narwhal_qa_studies: synthetic correctness bench."""
    return _finite_blob(narwhal_qa_studies.bench_narwhal_qa_studies(seed))


def bench_orca_qa_studies_family(seed: int = _SEED + 3):
    """orca_qa_studies: synthetic correctness bench."""
    return _finite_blob(orca_qa_studies.bench_orca_qa_studies(seed))


def bench_otter_qa_studies_family(seed: int = _SEED + 4):
    """otter_qa_studies: synthetic correctness bench."""
    return _finite_blob(otter_qa_studies.bench_otter_qa_studies(seed))


def bench_walrus_qa_studies_family(seed: int = _SEED + 5):
    """walrus_qa_studies: synthetic correctness bench."""
    return _finite_blob(walrus_qa_studies.bench_walrus_qa_studies(seed))
