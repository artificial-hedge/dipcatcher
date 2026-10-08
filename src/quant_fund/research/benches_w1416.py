"""Wave-1416 bench adapters: financial-NLP canon (SYNTHETIC only)."""

from quant_fund.models import (
    analyst_qa_studies,
    audit_qa_studies,
    bank_qa_studies,
    broker_qa_studies,
    credit_qa_studies,
    earnings_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14160


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


def bench_analyst_qa_studies_family(seed: int = _SEED + 0):
    """analyst_qa_studies: synthetic correctness bench."""
    return _finite_blob(analyst_qa_studies.bench_analyst_qa_studies(seed))


def bench_audit_qa_studies_family(seed: int = _SEED + 1):
    """audit_qa_studies: synthetic correctness bench."""
    return _finite_blob(audit_qa_studies.bench_audit_qa_studies(seed))


def bench_bank_qa_studies_family(seed: int = _SEED + 2):
    """bank_qa_studies: synthetic correctness bench."""
    return _finite_blob(bank_qa_studies.bench_bank_qa_studies(seed))


def bench_broker_qa_studies_family(seed: int = _SEED + 3):
    """broker_qa_studies: synthetic correctness bench."""
    return _finite_blob(broker_qa_studies.bench_broker_qa_studies(seed))


def bench_credit_qa_studies_family(seed: int = _SEED + 4):
    """credit_qa_studies: synthetic correctness bench."""
    return _finite_blob(credit_qa_studies.bench_credit_qa_studies(seed))


def bench_earnings_qa_studies_family(seed: int = _SEED + 5):
    """earnings_qa_studies: synthetic correctness bench."""
    return _finite_blob(earnings_qa_studies.bench_earnings_qa_studies(seed))
