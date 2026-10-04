"""Wave-1882 bench adapters: numidian-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ammed_qa_studies,
    laadas_qa_studies,
    langomed_qa_studies,
    mezzen_qa_studies,
    segimer_qa_studies,
    vercina_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18820


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ammed_qa_studies_family(seed: int = _SEED + 0):
    """ammed_qa_studies: synthetic correctness bench."""
    return _finite_blob(ammed_qa_studies.bench_ammed_qa_studies(seed))


def bench_laadas_qa_studies_family(seed: int = _SEED + 1):
    """laadas_qa_studies: synthetic correctness bench."""
    return _finite_blob(laadas_qa_studies.bench_laadas_qa_studies(seed))


def bench_langomed_qa_studies_family(seed: int = _SEED + 2):
    """langomed_qa_studies: synthetic correctness bench."""
    return _finite_blob(langomed_qa_studies.bench_langomed_qa_studies(seed))


def bench_mezzen_qa_studies_family(seed: int = _SEED + 3):
    """mezzen_qa_studies: synthetic correctness bench."""
    return _finite_blob(mezzen_qa_studies.bench_mezzen_qa_studies(seed))


def bench_segimer_qa_studies_family(seed: int = _SEED + 4):
    """segimer_qa_studies: synthetic correctness bench."""
    return _finite_blob(segimer_qa_studies.bench_segimer_qa_studies(seed))


def bench_vercina_qa_studies_family(seed: int = _SEED + 5):
    """vercina_qa_studies: synthetic correctness bench."""
    return _finite_blob(vercina_qa_studies.bench_vercina_qa_studies(seed))
