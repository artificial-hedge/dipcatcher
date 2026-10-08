"""Wave-1525 bench adapters: moss canon (SYNTHETIC only)."""

from quant_fund.models import (
    clubmoss_qa_studies,
    haircap_qa_studies,
    hornwort_qa_studies,
    liverwort_qa_studies,
    quillwort_qa_studies,
    sphagnum_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15250


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


def bench_clubmoss_qa_studies_family(seed: int = _SEED + 0):
    """clubmoss_qa_studies: synthetic correctness bench."""
    return _finite_blob(clubmoss_qa_studies.bench_clubmoss_qa_studies(seed))


def bench_haircap_qa_studies_family(seed: int = _SEED + 1):
    """haircap_qa_studies: synthetic correctness bench."""
    return _finite_blob(haircap_qa_studies.bench_haircap_qa_studies(seed))


def bench_hornwort_qa_studies_family(seed: int = _SEED + 2):
    """hornwort_qa_studies: synthetic correctness bench."""
    return _finite_blob(hornwort_qa_studies.bench_hornwort_qa_studies(seed))


def bench_liverwort_qa_studies_family(seed: int = _SEED + 3):
    """liverwort_qa_studies: synthetic correctness bench."""
    return _finite_blob(liverwort_qa_studies.bench_liverwort_qa_studies(seed))


def bench_quillwort_qa_studies_family(seed: int = _SEED + 4):
    """quillwort_qa_studies: synthetic correctness bench."""
    return _finite_blob(quillwort_qa_studies.bench_quillwort_qa_studies(seed))


def bench_sphagnum_qa_studies_family(seed: int = _SEED + 5):
    """sphagnum_qa_studies: synthetic correctness bench."""
    return _finite_blob(sphagnum_qa_studies.bench_sphagnum_qa_studies(seed))
