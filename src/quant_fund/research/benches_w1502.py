"""Wave-1502 bench adapters: neotropical-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    anteater_qa_studies,
    coatimundi_qa_studies,
    kinkajou_qa_studies,
    opossum_qa_studies,
    paca_qa_studies,
    tamandua_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15020


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anteater_qa_studies_family(seed: int = _SEED + 0):
    """anteater_qa_studies: synthetic correctness bench."""
    return _finite_blob(anteater_qa_studies.bench_anteater_qa_studies(seed))


def bench_coatimundi_qa_studies_family(seed: int = _SEED + 1):
    """coatimundi_qa_studies: synthetic correctness bench."""
    return _finite_blob(coatimundi_qa_studies.bench_coatimundi_qa_studies(seed))


def bench_kinkajou_qa_studies_family(seed: int = _SEED + 2):
    """kinkajou_qa_studies: synthetic correctness bench."""
    return _finite_blob(kinkajou_qa_studies.bench_kinkajou_qa_studies(seed))


def bench_opossum_qa_studies_family(seed: int = _SEED + 3):
    """opossum_qa_studies: synthetic correctness bench."""
    return _finite_blob(opossum_qa_studies.bench_opossum_qa_studies(seed))


def bench_paca_qa_studies_family(seed: int = _SEED + 4):
    """paca_qa_studies: synthetic correctness bench."""
    return _finite_blob(paca_qa_studies.bench_paca_qa_studies(seed))


def bench_tamandua_qa_studies_family(seed: int = _SEED + 5):
    """tamandua_qa_studies: synthetic correctness bench."""
    return _finite_blob(tamandua_qa_studies.bench_tamandua_qa_studies(seed))
