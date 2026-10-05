"""Wave-1945 bench adapters: oceania-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    adaro_qa_studies,
    kaiaimunu_qa_studies,
    masalai_qa_studies,
    pukaua_qa_studies,
    sanguma_qa_studies,
    tambaran_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19450


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_adaro_qa_studies_family(seed: int = _SEED + 0):
    """adaro_qa_studies: synthetic correctness bench."""
    return _finite_blob(adaro_qa_studies.bench_adaro_qa_studies(seed))


def bench_kaiaimunu_qa_studies_family(seed: int = _SEED + 1):
    """kaiaimunu_qa_studies: synthetic correctness bench."""
    return _finite_blob(kaiaimunu_qa_studies.bench_kaiaimunu_qa_studies(seed))


def bench_masalai_qa_studies_family(seed: int = _SEED + 2):
    """masalai_qa_studies: synthetic correctness bench."""
    return _finite_blob(masalai_qa_studies.bench_masalai_qa_studies(seed))


def bench_pukaua_qa_studies_family(seed: int = _SEED + 3):
    """pukaua_qa_studies: synthetic correctness bench."""
    return _finite_blob(pukaua_qa_studies.bench_pukaua_qa_studies(seed))


def bench_sanguma_qa_studies_family(seed: int = _SEED + 4):
    """sanguma_qa_studies: synthetic correctness bench."""
    return _finite_blob(sanguma_qa_studies.bench_sanguma_qa_studies(seed))


def bench_tambaran_qa_studies_family(seed: int = _SEED + 5):
    """tambaran_qa_studies: synthetic correctness bench."""
    return _finite_blob(tambaran_qa_studies.bench_tambaran_qa_studies(seed))
