"""Wave-1412 bench adapters: emotion-affect canon (SYNTHETIC only)."""

from quant_fund.models import (
    affect_qa_studies,
    anger_qa_studies,
    comfort_qa_studies,
    distress_qa_studies,
    emotion_qa_studies,
    empathy_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14120


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


def bench_affect_qa_studies_family(seed: int = _SEED + 0):
    """affect_qa_studies: synthetic correctness bench."""
    return _finite_blob(affect_qa_studies.bench_affect_qa_studies(seed))


def bench_anger_qa_studies_family(seed: int = _SEED + 1):
    """anger_qa_studies: synthetic correctness bench."""
    return _finite_blob(anger_qa_studies.bench_anger_qa_studies(seed))


def bench_comfort_qa_studies_family(seed: int = _SEED + 2):
    """comfort_qa_studies: synthetic correctness bench."""
    return _finite_blob(comfort_qa_studies.bench_comfort_qa_studies(seed))


def bench_distress_qa_studies_family(seed: int = _SEED + 3):
    """distress_qa_studies: synthetic correctness bench."""
    return _finite_blob(distress_qa_studies.bench_distress_qa_studies(seed))


def bench_emotion_qa_studies_family(seed: int = _SEED + 4):
    """emotion_qa_studies: synthetic correctness bench."""
    return _finite_blob(emotion_qa_studies.bench_emotion_qa_studies(seed))


def bench_empathy_qa_studies_family(seed: int = _SEED + 5):
    """empathy_qa_studies: synthetic correctness bench."""
    return _finite_blob(empathy_qa_studies.bench_empathy_qa_studies(seed))
