"""Wave-1602 bench adapters: fossorial canon (SYNTHETIC only)."""

from quant_fund.models import (
    desman_qa_studies,
    marsupial_mole_qa_studies,
    moles_lite_qa_studies,
    monotreme_qa_studies,
    moonrat_qa_studies,
    sengi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16020


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_desman_qa_studies_family(seed: int = _SEED + 0):
    """desman_qa_studies: synthetic correctness bench."""
    return _finite_blob(desman_qa_studies.bench_desman_qa_studies(seed))


def bench_marsupial_mole_qa_studies_family(seed: int = _SEED + 1):
    """marsupial_mole_qa_studies: synthetic correctness bench."""
    return _finite_blob(marsupial_mole_qa_studies.bench_marsupial_mole_qa_studies(seed))


def bench_moles_lite_qa_studies_family(seed: int = _SEED + 2):
    """moles_lite_qa_studies: synthetic correctness bench."""
    return _finite_blob(moles_lite_qa_studies.bench_moles_lite_qa_studies(seed))


def bench_monotreme_qa_studies_family(seed: int = _SEED + 3):
    """monotreme_qa_studies: synthetic correctness bench."""
    return _finite_blob(monotreme_qa_studies.bench_monotreme_qa_studies(seed))


def bench_moonrat_qa_studies_family(seed: int = _SEED + 4):
    """moonrat_qa_studies: synthetic correctness bench."""
    return _finite_blob(moonrat_qa_studies.bench_moonrat_qa_studies(seed))


def bench_sengi_qa_studies_family(seed: int = _SEED + 5):
    """sengi_qa_studies: synthetic correctness bench."""
    return _finite_blob(sengi_qa_studies.bench_sengi_qa_studies(seed))
