"""Wave-1338 bench adapters: math-reasoning-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    arith_qa_studies,
    gsm_hard_studies,
    math_reason_studies,
    mini_f2f_studies,
    proof_pile_studies,
    theorem_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13380


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


def bench_arith_qa_studies_family(seed: int = _SEED + 0):
    """arith_qa_studies: synthetic correctness bench."""
    return _finite_blob(arith_qa_studies.bench_arith_qa_studies(seed))


def bench_gsm_hard_studies_family(seed: int = _SEED + 1):
    """gsm_hard_studies: synthetic correctness bench."""
    return _finite_blob(gsm_hard_studies.bench_gsm_hard_studies(seed))


def bench_math_reason_studies_family(seed: int = _SEED + 2):
    """math_reason_studies: synthetic correctness bench."""
    return _finite_blob(math_reason_studies.bench_math_reason_studies(seed))


def bench_mini_f2f_studies_family(seed: int = _SEED + 3):
    """mini_f2f_studies: synthetic correctness bench."""
    return _finite_blob(mini_f2f_studies.bench_mini_f2f_studies(seed))


def bench_proof_pile_studies_family(seed: int = _SEED + 4):
    """proof_pile_studies: synthetic correctness bench."""
    return _finite_blob(proof_pile_studies.bench_proof_pile_studies(seed))


def bench_theorem_qa_studies_family(seed: int = _SEED + 5):
    """theorem_qa_studies: synthetic correctness bench."""
    return _finite_blob(theorem_qa_studies.bench_theorem_qa_studies(seed))
