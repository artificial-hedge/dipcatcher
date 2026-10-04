"""Wave-1733 bench adapters: irish-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    aengus_qa_studies,
    brigid_qa_studies,
    dagda_qa_studies,
    lugh_qa_studies,
    morrigan_qa_studies,
    nuada_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17330


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aengus_qa_studies_family(seed: int = _SEED + 0):
    """aengus_qa_studies: synthetic correctness bench."""
    return _finite_blob(aengus_qa_studies.bench_aengus_qa_studies(seed))


def bench_brigid_qa_studies_family(seed: int = _SEED + 1):
    """brigid_qa_studies: synthetic correctness bench."""
    return _finite_blob(brigid_qa_studies.bench_brigid_qa_studies(seed))


def bench_dagda_qa_studies_family(seed: int = _SEED + 2):
    """dagda_qa_studies: synthetic correctness bench."""
    return _finite_blob(dagda_qa_studies.bench_dagda_qa_studies(seed))


def bench_lugh_qa_studies_family(seed: int = _SEED + 3):
    """lugh_qa_studies: synthetic correctness bench."""
    return _finite_blob(lugh_qa_studies.bench_lugh_qa_studies(seed))


def bench_morrigan_qa_studies_family(seed: int = _SEED + 4):
    """morrigan_qa_studies: synthetic correctness bench."""
    return _finite_blob(morrigan_qa_studies.bench_morrigan_qa_studies(seed))


def bench_nuada_qa_studies_family(seed: int = _SEED + 5):
    """nuada_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuada_qa_studies.bench_nuada_qa_studies(seed))
