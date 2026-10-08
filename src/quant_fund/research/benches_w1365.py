"""Wave-1365 bench adapters: summarization canon (SYNTHETIC only)."""

from quant_fund.models import (
    arxiv_sum_studies,
    cnn_dailymail_studies,
    dialogsum_lite_studies,
    multi_news_studies,
    pubmed_sum_studies,
    samsum_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13650


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


def bench_arxiv_sum_studies_family(seed: int = _SEED + 0):
    """arxiv_sum_studies: synthetic correctness bench."""
    return _finite_blob(arxiv_sum_studies.bench_arxiv_sum_studies(seed))


def bench_cnn_dailymail_studies_family(seed: int = _SEED + 1):
    """cnn_dailymail_studies: synthetic correctness bench."""
    return _finite_blob(cnn_dailymail_studies.bench_cnn_dailymail_studies(seed))


def bench_dialogsum_lite_studies_family(seed: int = _SEED + 2):
    """dialogsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(dialogsum_lite_studies.bench_dialogsum_lite_studies(seed))


def bench_multi_news_studies_family(seed: int = _SEED + 3):
    """multi_news_studies: synthetic correctness bench."""
    return _finite_blob(multi_news_studies.bench_multi_news_studies(seed))


def bench_pubmed_sum_studies_family(seed: int = _SEED + 4):
    """pubmed_sum_studies: synthetic correctness bench."""
    return _finite_blob(pubmed_sum_studies.bench_pubmed_sum_studies(seed))


def bench_samsum_lite_studies_family(seed: int = _SEED + 5):
    """samsum_lite_studies: synthetic correctness bench."""
    return _finite_blob(samsum_lite_studies.bench_samsum_lite_studies(seed))
