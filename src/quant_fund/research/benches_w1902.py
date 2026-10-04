"""Wave-1902 bench adapters: yokai-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ao_andon_qa_studies,
    hannya_oni_qa_studies,
    kamaitachi_qa_studies,
    nurarihyon_qa_studies,
    shuten_doji_qa_studies,
    tsuchigumo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19020


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ao_andon_qa_studies_family(seed: int = _SEED + 0):
    """ao_andon_qa_studies: synthetic correctness bench."""
    return _finite_blob(ao_andon_qa_studies.bench_ao_andon_qa_studies(seed))


def bench_hannya_oni_qa_studies_family(seed: int = _SEED + 1):
    """hannya_oni_qa_studies: synthetic correctness bench."""
    return _finite_blob(hannya_oni_qa_studies.bench_hannya_oni_qa_studies(seed))


def bench_kamaitachi_qa_studies_family(seed: int = _SEED + 2):
    """kamaitachi_qa_studies: synthetic correctness bench."""
    return _finite_blob(kamaitachi_qa_studies.bench_kamaitachi_qa_studies(seed))


def bench_nurarihyon_qa_studies_family(seed: int = _SEED + 3):
    """nurarihyon_qa_studies: synthetic correctness bench."""
    return _finite_blob(nurarihyon_qa_studies.bench_nurarihyon_qa_studies(seed))


def bench_shuten_doji_qa_studies_family(seed: int = _SEED + 4):
    """shuten_doji_qa_studies: synthetic correctness bench."""
    return _finite_blob(shuten_doji_qa_studies.bench_shuten_doji_qa_studies(seed))


def bench_tsuchigumo_qa_studies_family(seed: int = _SEED + 5):
    """tsuchigumo_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsuchigumo_qa_studies.bench_tsuchigumo_qa_studies(seed))
