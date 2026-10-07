"""Wave-1570 bench adapters: crab canon (SYNTHETIC only)."""

from quant_fund.models import (
    fiddler_crab_qa_studies,
    ghost_crab_qa_studies,
    horseshoe_qa_studies,
    mud_crab_qa_studies,
    porcelain_qa_studies,
    spider_crab_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15700


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_fiddler_crab_qa_studies_family(seed: int = _SEED + 0):
    """fiddler_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(fiddler_crab_qa_studies.bench_fiddler_crab_qa_studies(seed))


def bench_ghost_crab_qa_studies_family(seed: int = _SEED + 1):
    """ghost_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(ghost_crab_qa_studies.bench_ghost_crab_qa_studies(seed))


def bench_horseshoe_qa_studies_family(seed: int = _SEED + 2):
    """horseshoe_qa_studies: synthetic correctness bench."""
    return _finite_blob(horseshoe_qa_studies.bench_horseshoe_qa_studies(seed))


def bench_mud_crab_qa_studies_family(seed: int = _SEED + 3):
    """mud_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(mud_crab_qa_studies.bench_mud_crab_qa_studies(seed))


def bench_porcelain_qa_studies_family(seed: int = _SEED + 4):
    """porcelain_qa_studies: synthetic correctness bench."""
    return _finite_blob(porcelain_qa_studies.bench_porcelain_qa_studies(seed))


def bench_spider_crab_qa_studies_family(seed: int = _SEED + 5):
    """spider_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(spider_crab_qa_studies.bench_spider_crab_qa_studies(seed))
