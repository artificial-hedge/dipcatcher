"""Wave-1859 bench adapters: welsh-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    beli_qa_studies,
    cassivellaunus_qa_studies,
    llefelys_qa_studies,
    manawydan_qa_studies,
    matholwch_qa_studies,
    pwll_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18590


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_beli_qa_studies_family(seed: int = _SEED + 0):
    """beli_qa_studies: synthetic correctness bench."""
    return _finite_blob(beli_qa_studies.bench_beli_qa_studies(seed))


def bench_cassivellaunus_qa_studies_family(seed: int = _SEED + 1):
    """cassivellaunus_qa_studies: synthetic correctness bench."""
    return _finite_blob(cassivellaunus_qa_studies.bench_cassivellaunus_qa_studies(seed))


def bench_llefelys_qa_studies_family(seed: int = _SEED + 2):
    """llefelys_qa_studies: synthetic correctness bench."""
    return _finite_blob(llefelys_qa_studies.bench_llefelys_qa_studies(seed))


def bench_manawydan_qa_studies_family(seed: int = _SEED + 3):
    """manawydan_qa_studies: synthetic correctness bench."""
    return _finite_blob(manawydan_qa_studies.bench_manawydan_qa_studies(seed))


def bench_matholwch_qa_studies_family(seed: int = _SEED + 4):
    """matholwch_qa_studies: synthetic correctness bench."""
    return _finite_blob(matholwch_qa_studies.bench_matholwch_qa_studies(seed))


def bench_pwll_qa_studies_family(seed: int = _SEED + 5):
    """pwll_qa_studies: synthetic correctness bench."""
    return _finite_blob(pwll_qa_studies.bench_pwll_qa_studies(seed))
