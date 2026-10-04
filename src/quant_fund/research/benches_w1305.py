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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
