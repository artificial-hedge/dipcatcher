"""Wave-1809 bench adapters: celtic-myth-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    brigid2_qa_studies,
    dagda2_qa_studies,
    lugh2_qa_studies,
    manannan2_qa_studies,
    nuada2_qa_studies,
    ogma2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18090


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_brigid2_qa_studies_family(seed: int = _SEED + 0):
    """brigid2_qa_studies: synthetic correctness bench."""
    return _finite_blob(brigid2_qa_studies.bench_brigid2_qa_studies(seed))


def bench_dagda2_qa_studies_family(seed: int = _SEED + 1):
    """dagda2_qa_studies: synthetic correctness bench."""
    return _finite_blob(dagda2_qa_studies.bench_dagda2_qa_studies(seed))


def bench_lugh2_qa_studies_family(seed: int = _SEED + 2):
    """lugh2_qa_studies: synthetic correctness bench."""
    return _finite_blob(lugh2_qa_studies.bench_lugh2_qa_studies(seed))


def bench_manannan2_qa_studies_family(seed: int = _SEED + 3):
    """manannan2_qa_studies: synthetic correctness bench."""
    return _finite_blob(manannan2_qa_studies.bench_manannan2_qa_studies(seed))


def bench_nuada2_qa_studies_family(seed: int = _SEED + 4):
    """nuada2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuada2_qa_studies.bench_nuada2_qa_studies(seed))


def bench_ogma2_qa_studies_family(seed: int = _SEED + 5):
    """ogma2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ogma2_qa_studies.bench_ogma2_qa_studies(seed))
