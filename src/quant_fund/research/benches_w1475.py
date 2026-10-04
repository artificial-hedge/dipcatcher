"""Wave-1475 bench adapters: invertebrate canon (SYNTHETIC only)."""

from quant_fund.models import (
    cicada_qa_studies,
    dragonfly_qa_studies,
    grasshopper_qa_studies,
    ladybug_qa_studies,
    mantis_qa_studies,
    scorpion_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14750


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cicada_qa_studies_family(seed: int = _SEED + 0):
    """cicada_qa_studies: synthetic correctness bench."""
    return _finite_blob(cicada_qa_studies.bench_cicada_qa_studies(seed))


def bench_dragonfly_qa_studies_family(seed: int = _SEED + 1):
    """dragonfly_qa_studies: synthetic correctness bench."""
    return _finite_blob(dragonfly_qa_studies.bench_dragonfly_qa_studies(seed))


def bench_grasshopper_qa_studies_family(seed: int = _SEED + 2):
    """grasshopper_qa_studies: synthetic correctness bench."""
    return _finite_blob(grasshopper_qa_studies.bench_grasshopper_qa_studies(seed))


def bench_ladybug_qa_studies_family(seed: int = _SEED + 3):
    """ladybug_qa_studies: synthetic correctness bench."""
    return _finite_blob(ladybug_qa_studies.bench_ladybug_qa_studies(seed))


def bench_mantis_qa_studies_family(seed: int = _SEED + 4):
    """mantis_qa_studies: synthetic correctness bench."""
    return _finite_blob(mantis_qa_studies.bench_mantis_qa_studies(seed))


def bench_scorpion_qa_studies_family(seed: int = _SEED + 5):
    """scorpion_qa_studies: synthetic correctness bench."""
    return _finite_blob(scorpion_qa_studies.bench_scorpion_qa_studies(seed))
