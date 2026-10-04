"""Wave-1755 bench adapters: norse-myth-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    dellingr_qa_studies,
    gna_qa_studies,
    jord_qa_studies,
    mani_qa_studies,
    sigyn_qa_studies,
    sol_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17550


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dellingr_qa_studies_family(seed: int = _SEED + 0):
    """dellingr_qa_studies: synthetic correctness bench."""
    return _finite_blob(dellingr_qa_studies.bench_dellingr_qa_studies(seed))


def bench_gna_qa_studies_family(seed: int = _SEED + 1):
    """gna_qa_studies: synthetic correctness bench."""
    return _finite_blob(gna_qa_studies.bench_gna_qa_studies(seed))


def bench_jord_qa_studies_family(seed: int = _SEED + 2):
    """jord_qa_studies: synthetic correctness bench."""
    return _finite_blob(jord_qa_studies.bench_jord_qa_studies(seed))


def bench_mani_qa_studies_family(seed: int = _SEED + 3):
    """mani_qa_studies: synthetic correctness bench."""
    return _finite_blob(mani_qa_studies.bench_mani_qa_studies(seed))


def bench_sigyn_qa_studies_family(seed: int = _SEED + 4):
    """sigyn_qa_studies: synthetic correctness bench."""
    return _finite_blob(sigyn_qa_studies.bench_sigyn_qa_studies(seed))


def bench_sol_qa_studies_family(seed: int = _SEED + 5):
    """sol_qa_studies: synthetic correctness bench."""
    return _finite_blob(sol_qa_studies.bench_sol_qa_studies(seed))
