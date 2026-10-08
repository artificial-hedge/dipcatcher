"""Wave-1305 bench adapters: reading-comprehension canon (SYNTHETIC only)."""

from quant_fund.models import (
    coqa_studies,
    drop_studies,
    hotpotqa_studies,
    nq_studies,
    squad_studies,
    triviaqa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13050


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


def bench_coqa_studies_family(seed: int = _SEED + 0):
    """coqa_studies: synthetic correctness bench."""
    return _finite_blob(coqa_studies.bench_coqa_studies(seed))


def bench_drop_studies_family(seed: int = _SEED + 1):
    """drop_studies: synthetic correctness bench."""
    return _finite_blob(drop_studies.bench_drop_studies(seed))


def bench_hotpotqa_studies_family(seed: int = _SEED + 2):
    """hotpotqa_studies: synthetic correctness bench."""
    return _finite_blob(hotpotqa_studies.bench_hotpotqa_studies(seed))


def bench_nq_studies_family(seed: int = _SEED + 3):
    """nq_studies: synthetic correctness bench."""
    return _finite_blob(nq_studies.bench_nq_studies(seed))


def bench_squad_studies_family(seed: int = _SEED + 4):
    """squad_studies: synthetic correctness bench."""
    return _finite_blob(squad_studies.bench_squad_studies(seed))


def bench_triviaqa_studies_family(seed: int = _SEED + 5):
    """triviaqa_studies: synthetic correctness bench."""
    return _finite_blob(triviaqa_studies.bench_triviaqa_studies(seed))
