"""Wave-1448 bench adapters: reptile canon (SYNTHETIC only)."""

from quant_fund.models import (
    cobra_qa_studies,
    frog_qa_studies,
    gecko_qa_studies,
    iguana_qa_studies,
    python_qa_studies,
    viper_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14480


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


def bench_cobra_qa_studies_family(seed: int = _SEED + 0):
    """cobra_qa_studies: synthetic correctness bench."""
    return _finite_blob(cobra_qa_studies.bench_cobra_qa_studies(seed))


def bench_frog_qa_studies_family(seed: int = _SEED + 1):
    """frog_qa_studies: synthetic correctness bench."""
    return _finite_blob(frog_qa_studies.bench_frog_qa_studies(seed))


def bench_gecko_qa_studies_family(seed: int = _SEED + 2):
    """gecko_qa_studies: synthetic correctness bench."""
    return _finite_blob(gecko_qa_studies.bench_gecko_qa_studies(seed))


def bench_iguana_qa_studies_family(seed: int = _SEED + 3):
    """iguana_qa_studies: synthetic correctness bench."""
    return _finite_blob(iguana_qa_studies.bench_iguana_qa_studies(seed))


def bench_python_qa_studies_family(seed: int = _SEED + 4):
    """python_qa_studies: synthetic correctness bench."""
    return _finite_blob(python_qa_studies.bench_python_qa_studies(seed))


def bench_viper_qa_studies_family(seed: int = _SEED + 5):
    """viper_qa_studies: synthetic correctness bench."""
    return _finite_blob(viper_qa_studies.bench_viper_qa_studies(seed))
