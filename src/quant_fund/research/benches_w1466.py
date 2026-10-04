"""Wave-1466 bench adapters: bedrock canon (SYNTHETIC only)."""

from quant_fund.models import (
    basalt_qa_studies,
    cathedral_qa_studies,
    chasm_qa_studies,
    crag_qa_studies,
    plateau_qa_studies,
    ravine_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14660


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_basalt_qa_studies_family(seed: int = _SEED + 0):
    """basalt_qa_studies: synthetic correctness bench."""
    return _finite_blob(basalt_qa_studies.bench_basalt_qa_studies(seed))


def bench_cathedral_qa_studies_family(seed: int = _SEED + 1):
    """cathedral_qa_studies: synthetic correctness bench."""
    return _finite_blob(cathedral_qa_studies.bench_cathedral_qa_studies(seed))


def bench_chasm_qa_studies_family(seed: int = _SEED + 2):
    """chasm_qa_studies: synthetic correctness bench."""
    return _finite_blob(chasm_qa_studies.bench_chasm_qa_studies(seed))


def bench_crag_qa_studies_family(seed: int = _SEED + 3):
    """crag_qa_studies: synthetic correctness bench."""
    return _finite_blob(crag_qa_studies.bench_crag_qa_studies(seed))


def bench_plateau_qa_studies_family(seed: int = _SEED + 4):
    """plateau_qa_studies: synthetic correctness bench."""
    return _finite_blob(plateau_qa_studies.bench_plateau_qa_studies(seed))


def bench_ravine_qa_studies_family(seed: int = _SEED + 5):
    """ravine_qa_studies: synthetic correctness bench."""
    return _finite_blob(ravine_qa_studies.bench_ravine_qa_studies(seed))
