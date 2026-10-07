"""Wave-1415 bench adapters: legal-regulatory canon (SYNTHETIC only)."""

from quant_fund.models import (
    case_qa_studies,
    clause_qa_studies,
    contract_qa_studies,
    lawqa_lite_studies,
    legal_qa_studies,
    statute_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14150


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


def bench_case_qa_studies_family(seed: int = _SEED + 0):
    """case_qa_studies: synthetic correctness bench."""
    return _finite_blob(case_qa_studies.bench_case_qa_studies(seed))


def bench_clause_qa_studies_family(seed: int = _SEED + 1):
    """clause_qa_studies: synthetic correctness bench."""
    return _finite_blob(clause_qa_studies.bench_clause_qa_studies(seed))


def bench_contract_qa_studies_family(seed: int = _SEED + 2):
    """contract_qa_studies: synthetic correctness bench."""
    return _finite_blob(contract_qa_studies.bench_contract_qa_studies(seed))


def bench_lawqa_lite_studies_family(seed: int = _SEED + 3):
    """lawqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(lawqa_lite_studies.bench_lawqa_lite_studies(seed))


def bench_legal_qa_studies_family(seed: int = _SEED + 4):
    """legal_qa_studies: synthetic correctness bench."""
    return _finite_blob(legal_qa_studies.bench_legal_qa_studies(seed))


def bench_statute_qa_studies_family(seed: int = _SEED + 5):
    """statute_qa_studies: synthetic correctness bench."""
    return _finite_blob(statute_qa_studies.bench_statute_qa_studies(seed))
