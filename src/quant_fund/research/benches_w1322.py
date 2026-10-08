"""Wave-1322 bench adapters: reasoning-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    logic_bench_studies,
    minif2f_studies,
    olympiad_bench_studies,
    putnam_studies,
    truthfulqa_studies,
    zebra_logic_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13220


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


def bench_logic_bench_studies_family(seed: int = _SEED + 0):
    """logic_bench_studies: synthetic correctness bench."""
    return _finite_blob(logic_bench_studies.bench_logic_bench_studies(seed))


def bench_minif2f_studies_family(seed: int = _SEED + 1):
    """minif2f_studies: synthetic correctness bench."""
    return _finite_blob(minif2f_studies.bench_minif2f_studies(seed))


def bench_olympiad_bench_studies_family(seed: int = _SEED + 2):
    """olympiad_bench_studies: synthetic correctness bench."""
    return _finite_blob(olympiad_bench_studies.bench_olympiad_bench_studies(seed))


def bench_putnam_studies_family(seed: int = _SEED + 3):
    """putnam_studies: synthetic correctness bench."""
    return _finite_blob(putnam_studies.bench_putnam_studies(seed))


def bench_truthfulqa_studies_family(seed: int = _SEED + 4):
    """truthfulqa_studies: synthetic correctness bench."""
    return _finite_blob(truthfulqa_studies.bench_truthfulqa_studies(seed))


def bench_zebra_logic_studies_family(seed: int = _SEED + 5):
    """zebra_logic_studies: synthetic correctness bench."""
    return _finite_blob(zebra_logic_studies.bench_zebra_logic_studies(seed))
