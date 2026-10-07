"""Wave-1584 bench adapters: small-mammal canon (SYNTHETIC only)."""

from quant_fund.models import (
    cottontail_qa_studies,
    hare_qa_studies,
    hedgehog_qa_studies,
    hyrax_qa_studies,
    jackrabbit_qa_studies,
    pika_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15840


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


def bench_cottontail_qa_studies_family(seed: int = _SEED + 0):
    """cottontail_qa_studies: synthetic correctness bench."""
    return _finite_blob(cottontail_qa_studies.bench_cottontail_qa_studies(seed))


def bench_hare_qa_studies_family(seed: int = _SEED + 1):
    """hare_qa_studies: synthetic correctness bench."""
    return _finite_blob(hare_qa_studies.bench_hare_qa_studies(seed))


def bench_hedgehog_qa_studies_family(seed: int = _SEED + 2):
    """hedgehog_qa_studies: synthetic correctness bench."""
    return _finite_blob(hedgehog_qa_studies.bench_hedgehog_qa_studies(seed))


def bench_hyrax_qa_studies_family(seed: int = _SEED + 3):
    """hyrax_qa_studies: synthetic correctness bench."""
    return _finite_blob(hyrax_qa_studies.bench_hyrax_qa_studies(seed))


def bench_jackrabbit_qa_studies_family(seed: int = _SEED + 4):
    """jackrabbit_qa_studies: synthetic correctness bench."""
    return _finite_blob(jackrabbit_qa_studies.bench_jackrabbit_qa_studies(seed))


def bench_pika_qa_studies_family(seed: int = _SEED + 5):
    """pika_qa_studies: synthetic correctness bench."""
    return _finite_blob(pika_qa_studies.bench_pika_qa_studies(seed))
