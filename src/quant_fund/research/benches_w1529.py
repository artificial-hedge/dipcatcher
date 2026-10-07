"""Wave-1529 bench adapters: gemstone canon (SYNTHETIC only)."""

from quant_fund.models import (
    aquamarine_qa_studies,
    garnet_qa_studies,
    opal_qa_studies,
    ruby_qa_studies,
    tanzanite_qa_studies,
    tourmaline_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15290


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


def bench_aquamarine_qa_studies_family(seed: int = _SEED + 0):
    """aquamarine_qa_studies: synthetic correctness bench."""
    return _finite_blob(aquamarine_qa_studies.bench_aquamarine_qa_studies(seed))


def bench_garnet_qa_studies_family(seed: int = _SEED + 1):
    """garnet_qa_studies: synthetic correctness bench."""
    return _finite_blob(garnet_qa_studies.bench_garnet_qa_studies(seed))


def bench_opal_qa_studies_family(seed: int = _SEED + 2):
    """opal_qa_studies: synthetic correctness bench."""
    return _finite_blob(opal_qa_studies.bench_opal_qa_studies(seed))


def bench_ruby_qa_studies_family(seed: int = _SEED + 3):
    """ruby_qa_studies: synthetic correctness bench."""
    return _finite_blob(ruby_qa_studies.bench_ruby_qa_studies(seed))


def bench_tanzanite_qa_studies_family(seed: int = _SEED + 4):
    """tanzanite_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanzanite_qa_studies.bench_tanzanite_qa_studies(seed))


def bench_tourmaline_qa_studies_family(seed: int = _SEED + 5):
    """tourmaline_qa_studies: synthetic correctness bench."""
    return _finite_blob(tourmaline_qa_studies.bench_tourmaline_qa_studies(seed))
