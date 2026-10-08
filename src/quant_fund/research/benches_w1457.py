"""Wave-1457 bench adapters: forest-mammal canon (SYNTHETIC only)."""

from quant_fund.models import (
    badger_qa_studies,
    beaver_qa_studies,
    bison_qa_studies,
    cougar_qa_studies,
    elk_qa_studies,
    lynx_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14570


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


def bench_badger_qa_studies_family(seed: int = _SEED + 0):
    """badger_qa_studies: synthetic correctness bench."""
    return _finite_blob(badger_qa_studies.bench_badger_qa_studies(seed))


def bench_beaver_qa_studies_family(seed: int = _SEED + 1):
    """beaver_qa_studies: synthetic correctness bench."""
    return _finite_blob(beaver_qa_studies.bench_beaver_qa_studies(seed))


def bench_bison_qa_studies_family(seed: int = _SEED + 2):
    """bison_qa_studies: synthetic correctness bench."""
    return _finite_blob(bison_qa_studies.bench_bison_qa_studies(seed))


def bench_cougar_qa_studies_family(seed: int = _SEED + 3):
    """cougar_qa_studies: synthetic correctness bench."""
    return _finite_blob(cougar_qa_studies.bench_cougar_qa_studies(seed))


def bench_elk_qa_studies_family(seed: int = _SEED + 4):
    """elk_qa_studies: synthetic correctness bench."""
    return _finite_blob(elk_qa_studies.bench_elk_qa_studies(seed))


def bench_lynx_qa_studies_family(seed: int = _SEED + 5):
    """lynx_qa_studies: synthetic correctness bench."""
    return _finite_blob(lynx_qa_studies.bench_lynx_qa_studies(seed))
