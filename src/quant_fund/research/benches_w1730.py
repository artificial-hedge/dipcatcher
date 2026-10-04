"""Wave-1730 bench adapters: turkic-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    aisit_qa_studies,
    bayna_qa_studies,
    kunkush_qa_studies,
    payna_qa_studies,
    taigan_qa_studies,
    yalyk_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17300


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aisit_qa_studies_family(seed: int = _SEED + 0):
    """aisit_qa_studies: synthetic correctness bench."""
    return _finite_blob(aisit_qa_studies.bench_aisit_qa_studies(seed))


def bench_bayna_qa_studies_family(seed: int = _SEED + 1):
    """bayna_qa_studies: synthetic correctness bench."""
    return _finite_blob(bayna_qa_studies.bench_bayna_qa_studies(seed))


def bench_kunkush_qa_studies_family(seed: int = _SEED + 2):
    """kunkush_qa_studies: synthetic correctness bench."""
    return _finite_blob(kunkush_qa_studies.bench_kunkush_qa_studies(seed))


def bench_payna_qa_studies_family(seed: int = _SEED + 3):
    """payna_qa_studies: synthetic correctness bench."""
    return _finite_blob(payna_qa_studies.bench_payna_qa_studies(seed))


def bench_taigan_qa_studies_family(seed: int = _SEED + 4):
    """taigan_qa_studies: synthetic correctness bench."""
    return _finite_blob(taigan_qa_studies.bench_taigan_qa_studies(seed))


def bench_yalyk_qa_studies_family(seed: int = _SEED + 5):
    """yalyk_qa_studies: synthetic correctness bench."""
    return _finite_blob(yalyk_qa_studies.bench_yalyk_qa_studies(seed))
