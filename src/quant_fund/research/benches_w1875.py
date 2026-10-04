"""Wave-1875 bench adapters: punic-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    abdastartus_qa_studies,
    bariha_qa_studies,
    bodastart_qa_studies,
    mider_qa_studies,
    reshef_qa_studies,
    safon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18750


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abdastartus_qa_studies_family(seed: int = _SEED + 0):
    """abdastartus_qa_studies: synthetic correctness bench."""
    return _finite_blob(abdastartus_qa_studies.bench_abdastartus_qa_studies(seed))


def bench_bariha_qa_studies_family(seed: int = _SEED + 1):
    """bariha_qa_studies: synthetic correctness bench."""
    return _finite_blob(bariha_qa_studies.bench_bariha_qa_studies(seed))


def bench_bodastart_qa_studies_family(seed: int = _SEED + 2):
    """bodastart_qa_studies: synthetic correctness bench."""
    return _finite_blob(bodastart_qa_studies.bench_bodastart_qa_studies(seed))


def bench_mider_qa_studies_family(seed: int = _SEED + 3):
    """mider_qa_studies: synthetic correctness bench."""
    return _finite_blob(mider_qa_studies.bench_mider_qa_studies(seed))


def bench_reshef_qa_studies_family(seed: int = _SEED + 4):
    """reshef_qa_studies: synthetic correctness bench."""
    return _finite_blob(reshef_qa_studies.bench_reshef_qa_studies(seed))


def bench_safon_qa_studies_family(seed: int = _SEED + 5):
    """safon_qa_studies: synthetic correctness bench."""
    return _finite_blob(safon_qa_studies.bench_safon_qa_studies(seed))
