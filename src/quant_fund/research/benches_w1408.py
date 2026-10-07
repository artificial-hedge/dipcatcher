"""Wave-1408 bench adapters: abductive-reasoning canon (SYNTHETIC only)."""

from quant_fund.models import (
    abduct_qa_studies,
    analogy_qa_studies,
    arct_lite_studies,
    entailment_qa_studies,
    fusion_qa_studies,
    proof_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14080


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


def bench_abduct_qa_studies_family(seed: int = _SEED + 0):
    """abduct_qa_studies: synthetic correctness bench."""
    return _finite_blob(abduct_qa_studies.bench_abduct_qa_studies(seed))


def bench_analogy_qa_studies_family(seed: int = _SEED + 1):
    """analogy_qa_studies: synthetic correctness bench."""
    return _finite_blob(analogy_qa_studies.bench_analogy_qa_studies(seed))


def bench_arct_lite_studies_family(seed: int = _SEED + 2):
    """arct_lite_studies: synthetic correctness bench."""
    return _finite_blob(arct_lite_studies.bench_arct_lite_studies(seed))


def bench_entailment_qa_studies_family(seed: int = _SEED + 3):
    """entailment_qa_studies: synthetic correctness bench."""
    return _finite_blob(entailment_qa_studies.bench_entailment_qa_studies(seed))


def bench_fusion_qa_studies_family(seed: int = _SEED + 4):
    """fusion_qa_studies: synthetic correctness bench."""
    return _finite_blob(fusion_qa_studies.bench_fusion_qa_studies(seed))


def bench_proof_qa_studies_family(seed: int = _SEED + 5):
    """proof_qa_studies: synthetic correctness bench."""
    return _finite_blob(proof_qa_studies.bench_proof_qa_studies(seed))
