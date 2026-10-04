"""Wave-1740 bench adapters: canaanite-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    anat_qa_studies,
    astarte_qa_studies,
    kothar_qa_studies,
    mot_qa_studies,
    resheph_qa_studies,
    yam_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17400


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anat_qa_studies_family(seed: int = _SEED + 0):
    """anat_qa_studies: synthetic correctness bench."""
    return _finite_blob(anat_qa_studies.bench_anat_qa_studies(seed))


def bench_astarte_qa_studies_family(seed: int = _SEED + 1):
    """astarte_qa_studies: synthetic correctness bench."""
    return _finite_blob(astarte_qa_studies.bench_astarte_qa_studies(seed))


def bench_kothar_qa_studies_family(seed: int = _SEED + 2):
    """kothar_qa_studies: synthetic correctness bench."""
    return _finite_blob(kothar_qa_studies.bench_kothar_qa_studies(seed))


def bench_mot_qa_studies_family(seed: int = _SEED + 3):
    """mot_qa_studies: synthetic correctness bench."""
    return _finite_blob(mot_qa_studies.bench_mot_qa_studies(seed))


def bench_resheph_qa_studies_family(seed: int = _SEED + 4):
    """resheph_qa_studies: synthetic correctness bench."""
    return _finite_blob(resheph_qa_studies.bench_resheph_qa_studies(seed))


def bench_yam_qa_studies_family(seed: int = _SEED + 5):
    """yam_qa_studies: synthetic correctness bench."""
    return _finite_blob(yam_qa_studies.bench_yam_qa_studies(seed))
