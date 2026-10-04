"""Wave-1554 bench adapters: spider canon (SYNTHETIC only)."""

from quant_fund.models import (
    black_widow_qa_studies,
    huntsman_qa_studies,
    jumping_spider_qa_studies,
    orb_weaver_qa_studies,
    tarantula_qa_studies,
    wolf_spider_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15540


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_black_widow_qa_studies_family(seed: int = _SEED + 0):
    """black_widow_qa_studies: synthetic correctness bench."""
    return _finite_blob(black_widow_qa_studies.bench_black_widow_qa_studies(seed))


def bench_huntsman_qa_studies_family(seed: int = _SEED + 1):
    """huntsman_qa_studies: synthetic correctness bench."""
    return _finite_blob(huntsman_qa_studies.bench_huntsman_qa_studies(seed))


def bench_jumping_spider_qa_studies_family(seed: int = _SEED + 2):
    """jumping_spider_qa_studies: synthetic correctness bench."""
    return _finite_blob(jumping_spider_qa_studies.bench_jumping_spider_qa_studies(seed))


def bench_orb_weaver_qa_studies_family(seed: int = _SEED + 3):
    """orb_weaver_qa_studies: synthetic correctness bench."""
    return _finite_blob(orb_weaver_qa_studies.bench_orb_weaver_qa_studies(seed))


def bench_tarantula_qa_studies_family(seed: int = _SEED + 4):
    """tarantula_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarantula_qa_studies.bench_tarantula_qa_studies(seed))


def bench_wolf_spider_qa_studies_family(seed: int = _SEED + 5):
    """wolf_spider_qa_studies: synthetic correctness bench."""
    return _finite_blob(wolf_spider_qa_studies.bench_wolf_spider_qa_studies(seed))
