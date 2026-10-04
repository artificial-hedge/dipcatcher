"""Wave-1686 bench adapters: african-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    abada_qa_studies,
    adze_qa_studies,
    ilomba_qa_studies,
    nbanda_qa_studies,
    ninki_qa_studies,
    okubi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16860


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abada_qa_studies_family(seed: int = _SEED + 0):
    """abada_qa_studies: synthetic correctness bench."""
    return _finite_blob(abada_qa_studies.bench_abada_qa_studies(seed))


def bench_adze_qa_studies_family(seed: int = _SEED + 1):
    """adze_qa_studies: synthetic correctness bench."""
    return _finite_blob(adze_qa_studies.bench_adze_qa_studies(seed))


def bench_ilomba_qa_studies_family(seed: int = _SEED + 2):
    """ilomba_qa_studies: synthetic correctness bench."""
    return _finite_blob(ilomba_qa_studies.bench_ilomba_qa_studies(seed))


def bench_nbanda_qa_studies_family(seed: int = _SEED + 3):
    """nbanda_qa_studies: synthetic correctness bench."""
    return _finite_blob(nbanda_qa_studies.bench_nbanda_qa_studies(seed))


def bench_ninki_qa_studies_family(seed: int = _SEED + 4):
    """ninki_qa_studies: synthetic correctness bench."""
    return _finite_blob(ninki_qa_studies.bench_ninki_qa_studies(seed))


def bench_okubi_qa_studies_family(seed: int = _SEED + 5):
    """okubi_qa_studies: synthetic correctness bench."""
    return _finite_blob(okubi_qa_studies.bench_okubi_qa_studies(seed))
