"""Wave-1435 bench adapters: anatomy canon (SYNTHETIC only)."""

from quant_fund.models import (
    blood_qa_studies,
    bone_qa_studies,
    brain_qa_studies,
    heart_qa_studies,
    muscle_qa_studies,
    nerve_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14350


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_blood_qa_studies_family(seed: int = _SEED + 0):
    """blood_qa_studies: synthetic correctness bench."""
    return _finite_blob(blood_qa_studies.bench_blood_qa_studies(seed))


def bench_bone_qa_studies_family(seed: int = _SEED + 1):
    """bone_qa_studies: synthetic correctness bench."""
    return _finite_blob(bone_qa_studies.bench_bone_qa_studies(seed))


def bench_brain_qa_studies_family(seed: int = _SEED + 2):
    """brain_qa_studies: synthetic correctness bench."""
    return _finite_blob(brain_qa_studies.bench_brain_qa_studies(seed))


def bench_heart_qa_studies_family(seed: int = _SEED + 3):
    """heart_qa_studies: synthetic correctness bench."""
    return _finite_blob(heart_qa_studies.bench_heart_qa_studies(seed))


def bench_muscle_qa_studies_family(seed: int = _SEED + 4):
    """muscle_qa_studies: synthetic correctness bench."""
    return _finite_blob(muscle_qa_studies.bench_muscle_qa_studies(seed))


def bench_nerve_qa_studies_family(seed: int = _SEED + 5):
    """nerve_qa_studies: synthetic correctness bench."""
    return _finite_blob(nerve_qa_studies.bench_nerve_qa_studies(seed))
