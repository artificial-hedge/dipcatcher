"""Wave-1728 bench adapters: hungarian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    boszorka_qa_studies,
    csaba_qa_studies,
    garabonci_qa_studies,
    isten_qa_studies,
    liderc_qa_studies,
    taltos_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17280


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_boszorka_qa_studies_family(seed: int = _SEED + 0):
    """boszorka_qa_studies: synthetic correctness bench."""
    return _finite_blob(boszorka_qa_studies.bench_boszorka_qa_studies(seed))


def bench_csaba_qa_studies_family(seed: int = _SEED + 1):
    """csaba_qa_studies: synthetic correctness bench."""
    return _finite_blob(csaba_qa_studies.bench_csaba_qa_studies(seed))


def bench_garabonci_qa_studies_family(seed: int = _SEED + 2):
    """garabonci_qa_studies: synthetic correctness bench."""
    return _finite_blob(garabonci_qa_studies.bench_garabonci_qa_studies(seed))


def bench_isten_qa_studies_family(seed: int = _SEED + 3):
    """isten_qa_studies: synthetic correctness bench."""
    return _finite_blob(isten_qa_studies.bench_isten_qa_studies(seed))


def bench_liderc_qa_studies_family(seed: int = _SEED + 4):
    """liderc_qa_studies: synthetic correctness bench."""
    return _finite_blob(liderc_qa_studies.bench_liderc_qa_studies(seed))


def bench_taltos_qa_studies_family(seed: int = _SEED + 5):
    """taltos_qa_studies: synthetic correctness bench."""
    return _finite_blob(taltos_qa_studies.bench_taltos_qa_studies(seed))
