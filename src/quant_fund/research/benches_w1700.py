"""Wave-1700 bench adapters: celtic-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    brigid_qa_studies,
    dagda_qa_studies,
    danu_qa_studies,
    lugh_qa_studies,
    manannan_qa_studies,
    morrigan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17000


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


def bench_brigid_qa_studies_family(seed: int = _SEED + 0):
    """brigid_qa_studies: synthetic correctness bench."""
    return _finite_blob(brigid_qa_studies.bench_brigid_qa_studies(seed))


def bench_dagda_qa_studies_family(seed: int = _SEED + 1):
    """dagda_qa_studies: synthetic correctness bench."""
    return _finite_blob(dagda_qa_studies.bench_dagda_qa_studies(seed))


def bench_danu_qa_studies_family(seed: int = _SEED + 2):
    """danu_qa_studies: synthetic correctness bench."""
    return _finite_blob(danu_qa_studies.bench_danu_qa_studies(seed))


def bench_lugh_qa_studies_family(seed: int = _SEED + 3):
    """lugh_qa_studies: synthetic correctness bench."""
    return _finite_blob(lugh_qa_studies.bench_lugh_qa_studies(seed))


def bench_manannan_qa_studies_family(seed: int = _SEED + 4):
    """manannan_qa_studies: synthetic correctness bench."""
    return _finite_blob(manannan_qa_studies.bench_manannan_qa_studies(seed))


def bench_morrigan_qa_studies_family(seed: int = _SEED + 5):
    """morrigan_qa_studies: synthetic correctness bench."""
    return _finite_blob(morrigan_qa_studies.bench_morrigan_qa_studies(seed))
