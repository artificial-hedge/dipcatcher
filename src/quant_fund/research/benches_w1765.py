"""Wave-1765 bench adapters: celtic-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    arianrhod_qa_studies,
    cerridwen_qa_studies,
    lugh_qa_studies,
    morrigan_qa_studies,
    nuada_qa_studies,
    rhiannon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17650


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arianrhod_qa_studies_family(seed: int = _SEED + 0):
    """arianrhod_qa_studies: synthetic correctness bench."""
    return _finite_blob(arianrhod_qa_studies.bench_arianrhod_qa_studies(seed))


def bench_cerridwen_qa_studies_family(seed: int = _SEED + 1):
    """cerridwen_qa_studies: synthetic correctness bench."""
    return _finite_blob(cerridwen_qa_studies.bench_cerridwen_qa_studies(seed))


def bench_lugh_qa_studies_family(seed: int = _SEED + 2):
    """lugh_qa_studies: synthetic correctness bench."""
    return _finite_blob(lugh_qa_studies.bench_lugh_qa_studies(seed))


def bench_morrigan_qa_studies_family(seed: int = _SEED + 3):
    """morrigan_qa_studies: synthetic correctness bench."""
    return _finite_blob(morrigan_qa_studies.bench_morrigan_qa_studies(seed))


def bench_nuada_qa_studies_family(seed: int = _SEED + 4):
    """nuada_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuada_qa_studies.bench_nuada_qa_studies(seed))


def bench_rhiannon_qa_studies_family(seed: int = _SEED + 5):
    """rhiannon_qa_studies: synthetic correctness bench."""
    return _finite_blob(rhiannon_qa_studies.bench_rhiannon_qa_studies(seed))
