"""Wave-1367 bench adapters: generation-metric canon (SYNTHETIC only)."""

from quant_fund.models import (
    bert_score_studies,
    bleu_rouge_studies,
    bleurt_lite_studies,
    comet_mt_studies,
    meteor_lite_studies,
    rouge_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13670


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bert_score_studies_family(seed: int = _SEED + 0):
    """bert_score_studies: synthetic correctness bench."""
    return _finite_blob(bert_score_studies.bench_bert_score_studies(seed))


def bench_bleu_rouge_studies_family(seed: int = _SEED + 1):
    """bleu_rouge_studies: synthetic correctness bench."""
    return _finite_blob(bleu_rouge_studies.bench_bleu_rouge_studies(seed))


def bench_bleurt_lite_studies_family(seed: int = _SEED + 2):
    """bleurt_lite_studies: synthetic correctness bench."""
    return _finite_blob(bleurt_lite_studies.bench_bleurt_lite_studies(seed))


def bench_comet_mt_studies_family(seed: int = _SEED + 3):
    """comet_mt_studies: synthetic correctness bench."""
    return _finite_blob(comet_mt_studies.bench_comet_mt_studies(seed))


def bench_meteor_lite_studies_family(seed: int = _SEED + 4):
    """meteor_lite_studies: synthetic correctness bench."""
    return _finite_blob(meteor_lite_studies.bench_meteor_lite_studies(seed))


def bench_rouge_lite_studies_family(seed: int = _SEED + 5):
    """rouge_lite_studies: synthetic correctness bench."""
    return _finite_blob(rouge_lite_studies.bench_rouge_lite_studies(seed))
