"""Wave-1332 bench adapters: eval-tooling canon (SYNTHETIC only)."""

from quant_fund.models import (
    decontaminate_studies,
    eval_bias_studies,
    fair_eval_studies,
    g_eval_studies,
    ngram_overlap_studies,
    pandalm_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13320


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_decontaminate_studies_family(seed: int = _SEED + 0):
    """decontaminate_studies: synthetic correctness bench."""
    return _finite_blob(decontaminate_studies.bench_decontaminate_studies(seed))


def bench_eval_bias_studies_family(seed: int = _SEED + 1):
    """eval_bias_studies: synthetic correctness bench."""
    return _finite_blob(eval_bias_studies.bench_eval_bias_studies(seed))


def bench_fair_eval_studies_family(seed: int = _SEED + 2):
    """fair_eval_studies: synthetic correctness bench."""
    return _finite_blob(fair_eval_studies.bench_fair_eval_studies(seed))


def bench_g_eval_studies_family(seed: int = _SEED + 3):
    """g_eval_studies: synthetic correctness bench."""
    return _finite_blob(g_eval_studies.bench_g_eval_studies(seed))


def bench_ngram_overlap_studies_family(seed: int = _SEED + 4):
    """ngram_overlap_studies: synthetic correctness bench."""
    return _finite_blob(ngram_overlap_studies.bench_ngram_overlap_studies(seed))


def bench_pandalm_studies_family(seed: int = _SEED + 5):
    """pandalm_studies: synthetic correctness bench."""
    return _finite_blob(pandalm_studies.bench_pandalm_studies(seed))
