"""Wave-1642 bench adapters: chimera canon (SYNTHETIC only)."""

from quant_fund.models import (
    basilisk_2_qa_studies,
    chimera_2_qa_studies,
    cockatrice_qa_studies,
    manticore_2_qa_studies,
    sphinx_2_qa_studies,
    wyvern_2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16420


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


def bench_basilisk_2_qa_studies_family(seed: int = _SEED + 0):
    """basilisk_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(basilisk_2_qa_studies.bench_basilisk_2_qa_studies(seed))


def bench_chimera_2_qa_studies_family(seed: int = _SEED + 1):
    """chimera_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(chimera_2_qa_studies.bench_chimera_2_qa_studies(seed))


def bench_cockatrice_qa_studies_family(seed: int = _SEED + 2):
    """cockatrice_qa_studies: synthetic correctness bench."""
    return _finite_blob(cockatrice_qa_studies.bench_cockatrice_qa_studies(seed))


def bench_manticore_2_qa_studies_family(seed: int = _SEED + 3):
    """manticore_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(manticore_2_qa_studies.bench_manticore_2_qa_studies(seed))


def bench_sphinx_2_qa_studies_family(seed: int = _SEED + 4):
    """sphinx_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sphinx_2_qa_studies.bench_sphinx_2_qa_studies(seed))


def bench_wyvern_2_qa_studies_family(seed: int = _SEED + 5):
    """wyvern_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(wyvern_2_qa_studies.bench_wyvern_2_qa_studies(seed))
