"""Wave-1638 bench adapters: yokai-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    akaname_qa_studies,
    hitodama_qa_studies,
    ittanmomen_qa_studies,
    nurikabe_qa_studies,
    shikigami_qa_studies,
    ubume_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16380


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_akaname_qa_studies_family(seed: int = _SEED + 0):
    """akaname_qa_studies: synthetic correctness bench."""
    return _finite_blob(akaname_qa_studies.bench_akaname_qa_studies(seed))


def bench_hitodama_qa_studies_family(seed: int = _SEED + 1):
    """hitodama_qa_studies: synthetic correctness bench."""
    return _finite_blob(hitodama_qa_studies.bench_hitodama_qa_studies(seed))


def bench_ittanmomen_qa_studies_family(seed: int = _SEED + 2):
    """ittanmomen_qa_studies: synthetic correctness bench."""
    return _finite_blob(ittanmomen_qa_studies.bench_ittanmomen_qa_studies(seed))


def bench_nurikabe_qa_studies_family(seed: int = _SEED + 3):
    """nurikabe_qa_studies: synthetic correctness bench."""
    return _finite_blob(nurikabe_qa_studies.bench_nurikabe_qa_studies(seed))


def bench_shikigami_qa_studies_family(seed: int = _SEED + 4):
    """shikigami_qa_studies: synthetic correctness bench."""
    return _finite_blob(shikigami_qa_studies.bench_shikigami_qa_studies(seed))


def bench_ubume_qa_studies_family(seed: int = _SEED + 5):
    """ubume_qa_studies: synthetic correctness bench."""
    return _finite_blob(ubume_qa_studies.bench_ubume_qa_studies(seed))
