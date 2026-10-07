"""Wave-1476 bench adapters: wildcat canon (SYNTHETIC only)."""

from quant_fund.models import (
    caracal_qa_studies,
    jaguarundi_qa_studies,
    margay_qa_studies,
    ocelot_qa_studies,
    puma_qa_studies,
    serval_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14760


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


def bench_caracal_qa_studies_family(seed: int = _SEED + 0):
    """caracal_qa_studies: synthetic correctness bench."""
    return _finite_blob(caracal_qa_studies.bench_caracal_qa_studies(seed))


def bench_jaguarundi_qa_studies_family(seed: int = _SEED + 1):
    """jaguarundi_qa_studies: synthetic correctness bench."""
    return _finite_blob(jaguarundi_qa_studies.bench_jaguarundi_qa_studies(seed))


def bench_margay_qa_studies_family(seed: int = _SEED + 2):
    """margay_qa_studies: synthetic correctness bench."""
    return _finite_blob(margay_qa_studies.bench_margay_qa_studies(seed))


def bench_ocelot_qa_studies_family(seed: int = _SEED + 3):
    """ocelot_qa_studies: synthetic correctness bench."""
    return _finite_blob(ocelot_qa_studies.bench_ocelot_qa_studies(seed))


def bench_puma_qa_studies_family(seed: int = _SEED + 4):
    """puma_qa_studies: synthetic correctness bench."""
    return _finite_blob(puma_qa_studies.bench_puma_qa_studies(seed))


def bench_serval_qa_studies_family(seed: int = _SEED + 5):
    """serval_qa_studies: synthetic correctness bench."""
    return _finite_blob(serval_qa_studies.bench_serval_qa_studies(seed))
