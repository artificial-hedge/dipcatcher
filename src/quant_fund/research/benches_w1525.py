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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
