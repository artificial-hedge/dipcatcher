"""Wave-1310 bench adapters: generation-quality canon (SYNTHETIC only)."""

from quant_fund.models import (
    alpaca_eval_studies,
    attribution_eval_studies,
    citation_eval_studies,
    diversity_eval_studies,
    factscore_studies,
    self_bleu_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13100


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


def bench_alpaca_eval_studies_family(seed: int = _SEED + 0):
    """alpaca_eval_studies: synthetic correctness bench."""
    return _finite_blob(alpaca_eval_studies.bench_alpaca_eval_studies(seed))


def bench_attribution_eval_studies_family(seed: int = _SEED + 1):
    """attribution_eval_studies: synthetic correctness bench."""
    return _finite_blob(attribution_eval_studies.bench_attribution_eval_studies(seed))


def bench_citation_eval_studies_family(seed: int = _SEED + 2):
    """citation_eval_studies: synthetic correctness bench."""
    return _finite_blob(citation_eval_studies.bench_citation_eval_studies(seed))


def bench_diversity_eval_studies_family(seed: int = _SEED + 3):
    """diversity_eval_studies: synthetic correctness bench."""
    return _finite_blob(diversity_eval_studies.bench_diversity_eval_studies(seed))


def bench_factscore_studies_family(seed: int = _SEED + 4):
    """factscore_studies: synthetic correctness bench."""
    return _finite_blob(factscore_studies.bench_factscore_studies(seed))


def bench_self_bleu_studies_family(seed: int = _SEED + 5):
    """self_bleu_studies: synthetic correctness bench."""
    return _finite_blob(self_bleu_studies.bench_self_bleu_studies(seed))
