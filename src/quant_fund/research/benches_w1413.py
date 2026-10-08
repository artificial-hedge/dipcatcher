"""Wave-1413 bench adapters: discourse-pragmatics canon (SYNTHETIC only)."""

from quant_fund.models import (
    anaphora_qa_studies,
    coherence_qa_studies,
    dialogue_act_studies,
    discourse_qa_studies,
    hedge_qa_studies,
    implicit_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14130


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


def bench_anaphora_qa_studies_family(seed: int = _SEED + 0):
    """anaphora_qa_studies: synthetic correctness bench."""
    return _finite_blob(anaphora_qa_studies.bench_anaphora_qa_studies(seed))


def bench_coherence_qa_studies_family(seed: int = _SEED + 1):
    """coherence_qa_studies: synthetic correctness bench."""
    return _finite_blob(coherence_qa_studies.bench_coherence_qa_studies(seed))


def bench_dialogue_act_studies_family(seed: int = _SEED + 2):
    """dialogue_act_studies: synthetic correctness bench."""
    return _finite_blob(dialogue_act_studies.bench_dialogue_act_studies(seed))


def bench_discourse_qa_studies_family(seed: int = _SEED + 3):
    """discourse_qa_studies: synthetic correctness bench."""
    return _finite_blob(discourse_qa_studies.bench_discourse_qa_studies(seed))


def bench_hedge_qa_studies_family(seed: int = _SEED + 4):
    """hedge_qa_studies: synthetic correctness bench."""
    return _finite_blob(hedge_qa_studies.bench_hedge_qa_studies(seed))


def bench_implicit_qa_studies_family(seed: int = _SEED + 5):
    """implicit_qa_studies: synthetic correctness bench."""
    return _finite_blob(implicit_qa_studies.bench_implicit_qa_studies(seed))
