"""Wave-1401 bench adapters: math-word-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    alg514_lite_studies,
    dolphin_lite_studies,
    draw_lite_studies,
    lila_lite_studies,
    math_doc_studies,
    math_eval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14010


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alg514_lite_studies_family(seed: int = _SEED + 0):
    """alg514_lite_studies: synthetic correctness bench."""
    return _finite_blob(alg514_lite_studies.bench_alg514_lite_studies(seed))


def bench_dolphin_lite_studies_family(seed: int = _SEED + 1):
    """dolphin_lite_studies: synthetic correctness bench."""
    return _finite_blob(dolphin_lite_studies.bench_dolphin_lite_studies(seed))


def bench_draw_lite_studies_family(seed: int = _SEED + 2):
    """draw_lite_studies: synthetic correctness bench."""
    return _finite_blob(draw_lite_studies.bench_draw_lite_studies(seed))


def bench_lila_lite_studies_family(seed: int = _SEED + 3):
    """lila_lite_studies: synthetic correctness bench."""
    return _finite_blob(lila_lite_studies.bench_lila_lite_studies(seed))


def bench_math_doc_studies_family(seed: int = _SEED + 4):
    """math_doc_studies: synthetic correctness bench."""
    return _finite_blob(math_doc_studies.bench_math_doc_studies(seed))


def bench_math_eval_studies_family(seed: int = _SEED + 5):
    """math_eval_studies: synthetic correctness bench."""
    return _finite_blob(math_eval_studies.bench_math_eval_studies(seed))
