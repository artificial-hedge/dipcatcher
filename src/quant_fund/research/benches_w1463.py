"""Wave-1463 bench adapters: meadow canon (SYNTHETIC only)."""

from quant_fund.models import (
    acorn_qa_studies,
    blossom_qa_studies,
    canopy_qa_studies,
    firefly_qa_studies,
    sprout_qa_studies,
    truffle_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14630


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_acorn_qa_studies_family(seed: int = _SEED + 0):
    """acorn_qa_studies: synthetic correctness bench."""
    return _finite_blob(acorn_qa_studies.bench_acorn_qa_studies(seed))


def bench_blossom_qa_studies_family(seed: int = _SEED + 1):
    """blossom_qa_studies: synthetic correctness bench."""
    return _finite_blob(blossom_qa_studies.bench_blossom_qa_studies(seed))


def bench_canopy_qa_studies_family(seed: int = _SEED + 2):
    """canopy_qa_studies: synthetic correctness bench."""
    return _finite_blob(canopy_qa_studies.bench_canopy_qa_studies(seed))


def bench_firefly_qa_studies_family(seed: int = _SEED + 3):
    """firefly_qa_studies: synthetic correctness bench."""
    return _finite_blob(firefly_qa_studies.bench_firefly_qa_studies(seed))


def bench_sprout_qa_studies_family(seed: int = _SEED + 4):
    """sprout_qa_studies: synthetic correctness bench."""
    return _finite_blob(sprout_qa_studies.bench_sprout_qa_studies(seed))


def bench_truffle_qa_studies_family(seed: int = _SEED + 5):
    """truffle_qa_studies: synthetic correctness bench."""
    return _finite_blob(truffle_qa_studies.bench_truffle_qa_studies(seed))
