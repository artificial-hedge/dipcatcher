"""Wave-1487 bench adapters: seabird-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    auk_qa_studies,
    fulmar_qa_studies,
    gull_qa_studies,
    jaeger_qa_studies,
    kittiwake_qa_studies,
    tropicbird_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14870


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_auk_qa_studies_family(seed: int = _SEED + 0):
    """auk_qa_studies: synthetic correctness bench."""
    return _finite_blob(auk_qa_studies.bench_auk_qa_studies(seed))


def bench_fulmar_qa_studies_family(seed: int = _SEED + 1):
    """fulmar_qa_studies: synthetic correctness bench."""
    return _finite_blob(fulmar_qa_studies.bench_fulmar_qa_studies(seed))


def bench_gull_qa_studies_family(seed: int = _SEED + 2):
    """gull_qa_studies: synthetic correctness bench."""
    return _finite_blob(gull_qa_studies.bench_gull_qa_studies(seed))


def bench_jaeger_qa_studies_family(seed: int = _SEED + 3):
    """jaeger_qa_studies: synthetic correctness bench."""
    return _finite_blob(jaeger_qa_studies.bench_jaeger_qa_studies(seed))


def bench_kittiwake_qa_studies_family(seed: int = _SEED + 4):
    """kittiwake_qa_studies: synthetic correctness bench."""
    return _finite_blob(kittiwake_qa_studies.bench_kittiwake_qa_studies(seed))


def bench_tropicbird_qa_studies_family(seed: int = _SEED + 5):
    """tropicbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(tropicbird_qa_studies.bench_tropicbird_qa_studies(seed))
