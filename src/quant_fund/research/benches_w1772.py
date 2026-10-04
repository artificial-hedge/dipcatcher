"""Wave-1772 bench adapters: greek-myth-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    hecate_qa_studies,
    helios_qa_studies,
    hypnos_qa_studies,
    selene_qa_studies,
    thanatos_qa_studies,
    zephyrus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17720


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hecate_qa_studies_family(seed: int = _SEED + 0):
    """hecate_qa_studies: synthetic correctness bench."""
    return _finite_blob(hecate_qa_studies.bench_hecate_qa_studies(seed))


def bench_helios_qa_studies_family(seed: int = _SEED + 1):
    """helios_qa_studies: synthetic correctness bench."""
    return _finite_blob(helios_qa_studies.bench_helios_qa_studies(seed))


def bench_hypnos_qa_studies_family(seed: int = _SEED + 2):
    """hypnos_qa_studies: synthetic correctness bench."""
    return _finite_blob(hypnos_qa_studies.bench_hypnos_qa_studies(seed))


def bench_selene_qa_studies_family(seed: int = _SEED + 3):
    """selene_qa_studies: synthetic correctness bench."""
    return _finite_blob(selene_qa_studies.bench_selene_qa_studies(seed))


def bench_thanatos_qa_studies_family(seed: int = _SEED + 4):
    """thanatos_qa_studies: synthetic correctness bench."""
    return _finite_blob(thanatos_qa_studies.bench_thanatos_qa_studies(seed))


def bench_zephyrus_qa_studies_family(seed: int = _SEED + 5):
    """zephyrus_qa_studies: synthetic correctness bench."""
    return _finite_blob(zephyrus_qa_studies.bench_zephyrus_qa_studies(seed))
