"""Wave-1907 bench adapters: african-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    aigamuxa_qa_studies,
    dodo_spirit_qa_studies,
    emere_qa_studies,
    kishi_demon_qa_studies,
    obayifo_qa_studies,
    ogboni_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19070


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aigamuxa_qa_studies_family(seed: int = _SEED + 0):
    """aigamuxa_qa_studies: synthetic correctness bench."""
    return _finite_blob(aigamuxa_qa_studies.bench_aigamuxa_qa_studies(seed))


def bench_dodo_spirit_qa_studies_family(seed: int = _SEED + 1):
    """dodo_spirit_qa_studies: synthetic correctness bench."""
    return _finite_blob(dodo_spirit_qa_studies.bench_dodo_spirit_qa_studies(seed))


def bench_emere_qa_studies_family(seed: int = _SEED + 2):
    """emere_qa_studies: synthetic correctness bench."""
    return _finite_blob(emere_qa_studies.bench_emere_qa_studies(seed))


def bench_kishi_demon_qa_studies_family(seed: int = _SEED + 3):
    """kishi_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(kishi_demon_qa_studies.bench_kishi_demon_qa_studies(seed))


def bench_obayifo_qa_studies_family(seed: int = _SEED + 4):
    """obayifo_qa_studies: synthetic correctness bench."""
    return _finite_blob(obayifo_qa_studies.bench_obayifo_qa_studies(seed))


def bench_ogboni_qa_studies_family(seed: int = _SEED + 5):
    """ogboni_qa_studies: synthetic correctness bench."""
    return _finite_blob(ogboni_qa_studies.bench_ogboni_qa_studies(seed))
