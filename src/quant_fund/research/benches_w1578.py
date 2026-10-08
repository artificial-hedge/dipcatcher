"""Wave-1578 bench adapters: ungulate canon (SYNTHETIC only)."""

from quant_fund.models import (
    gerenuk_qa_studies,
    markhor_qa_studies,
    nilgai_qa_studies,
    okapi_qa_studies,
    saiga_qa_studies,
    takin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15780


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


def bench_gerenuk_qa_studies_family(seed: int = _SEED + 0):
    """gerenuk_qa_studies: synthetic correctness bench."""
    return _finite_blob(gerenuk_qa_studies.bench_gerenuk_qa_studies(seed))


def bench_markhor_qa_studies_family(seed: int = _SEED + 1):
    """markhor_qa_studies: synthetic correctness bench."""
    return _finite_blob(markhor_qa_studies.bench_markhor_qa_studies(seed))


def bench_nilgai_qa_studies_family(seed: int = _SEED + 2):
    """nilgai_qa_studies: synthetic correctness bench."""
    return _finite_blob(nilgai_qa_studies.bench_nilgai_qa_studies(seed))


def bench_okapi_qa_studies_family(seed: int = _SEED + 3):
    """okapi_qa_studies: synthetic correctness bench."""
    return _finite_blob(okapi_qa_studies.bench_okapi_qa_studies(seed))


def bench_saiga_qa_studies_family(seed: int = _SEED + 4):
    """saiga_qa_studies: synthetic correctness bench."""
    return _finite_blob(saiga_qa_studies.bench_saiga_qa_studies(seed))


def bench_takin_qa_studies_family(seed: int = _SEED + 5):
    """takin_qa_studies: synthetic correctness bench."""
    return _finite_blob(takin_qa_studies.bench_takin_qa_studies(seed))
