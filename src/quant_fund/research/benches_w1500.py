"""Wave-1500 bench adapters: invertebrate-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    caddisfly_qa_studies,
    centipede_qa_studies,
    horntail_qa_studies,
    lacewing_qa_studies,
    millipede_qa_studies,
    spider_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15000


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_caddisfly_qa_studies_family(seed: int = _SEED + 0):
    """caddisfly_qa_studies: synthetic correctness bench."""
    return _finite_blob(caddisfly_qa_studies.bench_caddisfly_qa_studies(seed))


def bench_centipede_qa_studies_family(seed: int = _SEED + 1):
    """centipede_qa_studies: synthetic correctness bench."""
    return _finite_blob(centipede_qa_studies.bench_centipede_qa_studies(seed))


def bench_horntail_qa_studies_family(seed: int = _SEED + 2):
    """horntail_qa_studies: synthetic correctness bench."""
    return _finite_blob(horntail_qa_studies.bench_horntail_qa_studies(seed))


def bench_lacewing_qa_studies_family(seed: int = _SEED + 3):
    """lacewing_qa_studies: synthetic correctness bench."""
    return _finite_blob(lacewing_qa_studies.bench_lacewing_qa_studies(seed))


def bench_millipede_qa_studies_family(seed: int = _SEED + 4):
    """millipede_qa_studies: synthetic correctness bench."""
    return _finite_blob(millipede_qa_studies.bench_millipede_qa_studies(seed))


def bench_spider_qa_studies_family(seed: int = _SEED + 5):
    """spider_qa_studies: synthetic correctness bench."""
    return _finite_blob(spider_qa_studies.bench_spider_qa_studies(seed))
