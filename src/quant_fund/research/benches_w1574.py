"""Wave-1574 bench adapters: primate canon (SYNTHETIC only)."""

from quant_fund.models import (
    gibbon_qa_studies,
    langur_qa_studies,
    lemur_qa_studies,
    macaque_qa_studies,
    marmoset_qa_studies,
    tamarin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15740


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


def bench_gibbon_qa_studies_family(seed: int = _SEED + 0):
    """gibbon_qa_studies: synthetic correctness bench."""
    return _finite_blob(gibbon_qa_studies.bench_gibbon_qa_studies(seed))


def bench_langur_qa_studies_family(seed: int = _SEED + 1):
    """langur_qa_studies: synthetic correctness bench."""
    return _finite_blob(langur_qa_studies.bench_langur_qa_studies(seed))


def bench_lemur_qa_studies_family(seed: int = _SEED + 2):
    """lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(lemur_qa_studies.bench_lemur_qa_studies(seed))


def bench_macaque_qa_studies_family(seed: int = _SEED + 3):
    """macaque_qa_studies: synthetic correctness bench."""
    return _finite_blob(macaque_qa_studies.bench_macaque_qa_studies(seed))


def bench_marmoset_qa_studies_family(seed: int = _SEED + 4):
    """marmoset_qa_studies: synthetic correctness bench."""
    return _finite_blob(marmoset_qa_studies.bench_marmoset_qa_studies(seed))


def bench_tamarin_qa_studies_family(seed: int = _SEED + 5):
    """tamarin_qa_studies: synthetic correctness bench."""
    return _finite_blob(tamarin_qa_studies.bench_tamarin_qa_studies(seed))
