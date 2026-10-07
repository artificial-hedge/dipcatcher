"""Wave-1492 bench adapters: marsupial-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bilby_qa_studies,
    echidna_qa_studies,
    platypus_qa_studies,
    possum_qa_studies,
    quoll_qa_studies,
    thylacine_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14920


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


def bench_bilby_qa_studies_family(seed: int = _SEED + 0):
    """bilby_qa_studies: synthetic correctness bench."""
    return _finite_blob(bilby_qa_studies.bench_bilby_qa_studies(seed))


def bench_echidna_qa_studies_family(seed: int = _SEED + 1):
    """echidna_qa_studies: synthetic correctness bench."""
    return _finite_blob(echidna_qa_studies.bench_echidna_qa_studies(seed))


def bench_platypus_qa_studies_family(seed: int = _SEED + 2):
    """platypus_qa_studies: synthetic correctness bench."""
    return _finite_blob(platypus_qa_studies.bench_platypus_qa_studies(seed))


def bench_possum_qa_studies_family(seed: int = _SEED + 3):
    """possum_qa_studies: synthetic correctness bench."""
    return _finite_blob(possum_qa_studies.bench_possum_qa_studies(seed))


def bench_quoll_qa_studies_family(seed: int = _SEED + 4):
    """quoll_qa_studies: synthetic correctness bench."""
    return _finite_blob(quoll_qa_studies.bench_quoll_qa_studies(seed))


def bench_thylacine_qa_studies_family(seed: int = _SEED + 5):
    """thylacine_qa_studies: synthetic correctness bench."""
    return _finite_blob(thylacine_qa_studies.bench_thylacine_qa_studies(seed))
