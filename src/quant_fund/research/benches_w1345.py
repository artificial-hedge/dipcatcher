"""Wave-1345 bench adapters: open-domain-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    complex_qa_studies,
    entity_quests_studies,
    freebase_qa_studies,
    nq_open_studies,
    trivia_qa_studies,
    web_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13450


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


def bench_complex_qa_studies_family(seed: int = _SEED + 0):
    """complex_qa_studies: synthetic correctness bench."""
    return _finite_blob(complex_qa_studies.bench_complex_qa_studies(seed))


def bench_entity_quests_studies_family(seed: int = _SEED + 1):
    """entity_quests_studies: synthetic correctness bench."""
    return _finite_blob(entity_quests_studies.bench_entity_quests_studies(seed))


def bench_freebase_qa_studies_family(seed: int = _SEED + 2):
    """freebase_qa_studies: synthetic correctness bench."""
    return _finite_blob(freebase_qa_studies.bench_freebase_qa_studies(seed))


def bench_nq_open_studies_family(seed: int = _SEED + 3):
    """nq_open_studies: synthetic correctness bench."""
    return _finite_blob(nq_open_studies.bench_nq_open_studies(seed))


def bench_trivia_qa_studies_family(seed: int = _SEED + 4):
    """trivia_qa_studies: synthetic correctness bench."""
    return _finite_blob(trivia_qa_studies.bench_trivia_qa_studies(seed))


def bench_web_qa_studies_family(seed: int = _SEED + 5):
    """web_qa_studies: synthetic correctness bench."""
    return _finite_blob(web_qa_studies.bench_web_qa_studies(seed))
