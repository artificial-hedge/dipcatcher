"""Wave-1379 bench adapters: metric-exotics canon (SYNTHETIC only)."""

from quant_fund.models import (
    bary_score_studies,
    cider_lite_studies,
    gleu_lite_studies,
    kl_div_eval_studies,
    rouge_we_studies,
    wmt_metric_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13790


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bary_score_studies_family(seed: int = _SEED + 0):
    """bary_score_studies: synthetic correctness bench."""
    return _finite_blob(bary_score_studies.bench_bary_score_studies(seed))


def bench_cider_lite_studies_family(seed: int = _SEED + 1):
    """cider_lite_studies: synthetic correctness bench."""
    return _finite_blob(cider_lite_studies.bench_cider_lite_studies(seed))


def bench_gleu_lite_studies_family(seed: int = _SEED + 2):
    """gleu_lite_studies: synthetic correctness bench."""
    return _finite_blob(gleu_lite_studies.bench_gleu_lite_studies(seed))


def bench_kl_div_eval_studies_family(seed: int = _SEED + 3):
    """kl_div_eval_studies: synthetic correctness bench."""
    return _finite_blob(kl_div_eval_studies.bench_kl_div_eval_studies(seed))


def bench_rouge_we_studies_family(seed: int = _SEED + 4):
    """rouge_we_studies: synthetic correctness bench."""
    return _finite_blob(rouge_we_studies.bench_rouge_we_studies(seed))


def bench_wmt_metric_studies_family(seed: int = _SEED + 5):
    """wmt_metric_studies: synthetic correctness bench."""
    return _finite_blob(wmt_metric_studies.bench_wmt_metric_studies(seed))
