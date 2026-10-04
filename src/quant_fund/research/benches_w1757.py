"""Wave-1757 bench adapters: celtic-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    aisling_qa_studies,
    brigid_qa_studies,
    dagda_qa_studies,
    danu_qa_studies,
    manannan_qa_studies,
    morgen_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17570


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aisling_qa_studies_family(seed: int = _SEED + 0):
    """aisling_qa_studies: synthetic correctness bench."""
    return _finite_blob(aisling_qa_studies.bench_aisling_qa_studies(seed))


def bench_brigid_qa_studies_family(seed: int = _SEED + 1):
    """brigid_qa_studies: synthetic correctness bench."""
    return _finite_blob(brigid_qa_studies.bench_brigid_qa_studies(seed))


def bench_dagda_qa_studies_family(seed: int = _SEED + 2):
    """dagda_qa_studies: synthetic correctness bench."""
    return _finite_blob(dagda_qa_studies.bench_dagda_qa_studies(seed))


def bench_danu_qa_studies_family(seed: int = _SEED + 3):
    """danu_qa_studies: synthetic correctness bench."""
    return _finite_blob(danu_qa_studies.bench_danu_qa_studies(seed))


def bench_manannan_qa_studies_family(seed: int = _SEED + 4):
    """manannan_qa_studies: synthetic correctness bench."""
    return _finite_blob(manannan_qa_studies.bench_manannan_qa_studies(seed))


def bench_morgen_qa_studies_family(seed: int = _SEED + 5):
    """morgen_qa_studies: synthetic correctness bench."""
    return _finite_blob(morgen_qa_studies.bench_morgen_qa_studies(seed))
