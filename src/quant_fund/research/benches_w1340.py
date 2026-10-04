"""Wave-1340 bench adapters: science-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    arc_challenge_studies,
    bio_qa_studies,
    med_qa_studies,
    openbook_qa_studies,
    pubmed_qa_studies,
    sci_q_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13400


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arc_challenge_studies_family(seed: int = _SEED + 0):
    """arc_challenge_studies: synthetic correctness bench."""
    return _finite_blob(arc_challenge_studies.bench_arc_challenge_studies(seed))


def bench_bio_qa_studies_family(seed: int = _SEED + 1):
    """bio_qa_studies: synthetic correctness bench."""
    return _finite_blob(bio_qa_studies.bench_bio_qa_studies(seed))


def bench_med_qa_studies_family(seed: int = _SEED + 2):
    """med_qa_studies: synthetic correctness bench."""
    return _finite_blob(med_qa_studies.bench_med_qa_studies(seed))


def bench_openbook_qa_studies_family(seed: int = _SEED + 3):
    """openbook_qa_studies: synthetic correctness bench."""
    return _finite_blob(openbook_qa_studies.bench_openbook_qa_studies(seed))


def bench_pubmed_qa_studies_family(seed: int = _SEED + 4):
    """pubmed_qa_studies: synthetic correctness bench."""
    return _finite_blob(pubmed_qa_studies.bench_pubmed_qa_studies(seed))


def bench_sci_q_studies_family(seed: int = _SEED + 5):
    """sci_q_studies: synthetic correctness bench."""
    return _finite_blob(sci_q_studies.bench_sci_q_studies(seed))
