"""Wave-1575 bench adapters: rodent canon (SYNTHETIC only)."""

from quant_fund.models import (
    chinchilla_qa_studies,
    degu_qa_studies,
    gerbil_qa_studies,
    hamster_qa_studies,
    lemming_qa_studies,
    vole_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15750


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


def bench_chinchilla_qa_studies_family(seed: int = _SEED + 0):
    """chinchilla_qa_studies: synthetic correctness bench."""
    return _finite_blob(chinchilla_qa_studies.bench_chinchilla_qa_studies(seed))


def bench_degu_qa_studies_family(seed: int = _SEED + 1):
    """degu_qa_studies: synthetic correctness bench."""
    return _finite_blob(degu_qa_studies.bench_degu_qa_studies(seed))


def bench_gerbil_qa_studies_family(seed: int = _SEED + 2):
    """gerbil_qa_studies: synthetic correctness bench."""
    return _finite_blob(gerbil_qa_studies.bench_gerbil_qa_studies(seed))


def bench_hamster_qa_studies_family(seed: int = _SEED + 3):
    """hamster_qa_studies: synthetic correctness bench."""
    return _finite_blob(hamster_qa_studies.bench_hamster_qa_studies(seed))


def bench_lemming_qa_studies_family(seed: int = _SEED + 4):
    """lemming_qa_studies: synthetic correctness bench."""
    return _finite_blob(lemming_qa_studies.bench_lemming_qa_studies(seed))


def bench_vole_qa_studies_family(seed: int = _SEED + 5):
    """vole_qa_studies: synthetic correctness bench."""
    return _finite_blob(vole_qa_studies.bench_vole_qa_studies(seed))
