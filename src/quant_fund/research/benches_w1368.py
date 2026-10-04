"""Wave-1368 bench adapters: translation-metric canon (SYNTHETIC only)."""

from quant_fund.models import (
    chr_f_studies,
    mover_score_studies,
    nist_metric_studies,
    prism_mt_studies,
    sacrebleu_lite_studies,
    ter_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13680


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_chr_f_studies_family(seed: int = _SEED + 0):
    """chr_f_studies: synthetic correctness bench."""
    return _finite_blob(chr_f_studies.bench_chr_f_studies(seed))


def bench_mover_score_studies_family(seed: int = _SEED + 1):
    """mover_score_studies: synthetic correctness bench."""
    return _finite_blob(mover_score_studies.bench_mover_score_studies(seed))


def bench_nist_metric_studies_family(seed: int = _SEED + 2):
    """nist_metric_studies: synthetic correctness bench."""
    return _finite_blob(nist_metric_studies.bench_nist_metric_studies(seed))


def bench_prism_mt_studies_family(seed: int = _SEED + 3):
    """prism_mt_studies: synthetic correctness bench."""
    return _finite_blob(prism_mt_studies.bench_prism_mt_studies(seed))


def bench_sacrebleu_lite_studies_family(seed: int = _SEED + 4):
    """sacrebleu_lite_studies: synthetic correctness bench."""
    return _finite_blob(sacrebleu_lite_studies.bench_sacrebleu_lite_studies(seed))


def bench_ter_lite_studies_family(seed: int = _SEED + 5):
    """ter_lite_studies: synthetic correctness bench."""
    return _finite_blob(ter_lite_studies.bench_ter_lite_studies(seed))
