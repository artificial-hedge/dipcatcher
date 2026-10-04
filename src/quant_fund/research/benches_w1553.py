"""Wave-1553 bench adapters: frog canon (SYNTHETIC only)."""

from quant_fund.models import (
    dart_frog_qa_studies,
    horned_frog_qa_studies,
    leopard_frog_qa_studies,
    spring_peeper_qa_studies,
    treefrog_qa_studies,
    wood_frog_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15530


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dart_frog_qa_studies_family(seed: int = _SEED + 0):
    """dart_frog_qa_studies: synthetic correctness bench."""
    return _finite_blob(dart_frog_qa_studies.bench_dart_frog_qa_studies(seed))


def bench_horned_frog_qa_studies_family(seed: int = _SEED + 1):
    """horned_frog_qa_studies: synthetic correctness bench."""
    return _finite_blob(horned_frog_qa_studies.bench_horned_frog_qa_studies(seed))


def bench_leopard_frog_qa_studies_family(seed: int = _SEED + 2):
    """leopard_frog_qa_studies: synthetic correctness bench."""
    return _finite_blob(leopard_frog_qa_studies.bench_leopard_frog_qa_studies(seed))


def bench_spring_peeper_qa_studies_family(seed: int = _SEED + 3):
    """spring_peeper_qa_studies: synthetic correctness bench."""
    return _finite_blob(spring_peeper_qa_studies.bench_spring_peeper_qa_studies(seed))


def bench_treefrog_qa_studies_family(seed: int = _SEED + 4):
    """treefrog_qa_studies: synthetic correctness bench."""
    return _finite_blob(treefrog_qa_studies.bench_treefrog_qa_studies(seed))


def bench_wood_frog_qa_studies_family(seed: int = _SEED + 5):
    """wood_frog_qa_studies: synthetic correctness bench."""
    return _finite_blob(wood_frog_qa_studies.bench_wood_frog_qa_studies(seed))
