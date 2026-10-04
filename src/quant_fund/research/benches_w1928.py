"""Wave-1928 bench adapters: polynesian-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    kehua_qa_studies,
    maero_qa_studies,
    ngarara_qa_studies,
    patupaiarehe_qa_studies,
    ponaturi_qa_studies,
    taipo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19280


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_kehua_qa_studies_family(seed: int = _SEED + 0):
    """kehua_qa_studies: synthetic correctness bench."""
    return _finite_blob(kehua_qa_studies.bench_kehua_qa_studies(seed))


def bench_maero_qa_studies_family(seed: int = _SEED + 1):
    """maero_qa_studies: synthetic correctness bench."""
    return _finite_blob(maero_qa_studies.bench_maero_qa_studies(seed))


def bench_ngarara_qa_studies_family(seed: int = _SEED + 2):
    """ngarara_qa_studies: synthetic correctness bench."""
    return _finite_blob(ngarara_qa_studies.bench_ngarara_qa_studies(seed))


def bench_patupaiarehe_qa_studies_family(seed: int = _SEED + 3):
    """patupaiarehe_qa_studies: synthetic correctness bench."""
    return _finite_blob(patupaiarehe_qa_studies.bench_patupaiarehe_qa_studies(seed))


def bench_ponaturi_qa_studies_family(seed: int = _SEED + 4):
    """ponaturi_qa_studies: synthetic correctness bench."""
    return _finite_blob(ponaturi_qa_studies.bench_ponaturi_qa_studies(seed))


def bench_taipo_qa_studies_family(seed: int = _SEED + 5):
    """taipo_qa_studies: synthetic correctness bench."""
    return _finite_blob(taipo_qa_studies.bench_taipo_qa_studies(seed))
