"""Wave-1778 bench adapters: hindu-myth-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    hanuman_qa_studies,
    lakshmi_qa_studies,
    parvati_qa_studies,
    rama_qa_studies,
    sita_qa_studies,
    vishnu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17780


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hanuman_qa_studies_family(seed: int = _SEED + 0):
    """hanuman_qa_studies: synthetic correctness bench."""
    return _finite_blob(hanuman_qa_studies.bench_hanuman_qa_studies(seed))


def bench_lakshmi_qa_studies_family(seed: int = _SEED + 1):
    """lakshmi_qa_studies: synthetic correctness bench."""
    return _finite_blob(lakshmi_qa_studies.bench_lakshmi_qa_studies(seed))


def bench_parvati_qa_studies_family(seed: int = _SEED + 2):
    """parvati_qa_studies: synthetic correctness bench."""
    return _finite_blob(parvati_qa_studies.bench_parvati_qa_studies(seed))


def bench_rama_qa_studies_family(seed: int = _SEED + 3):
    """rama_qa_studies: synthetic correctness bench."""
    return _finite_blob(rama_qa_studies.bench_rama_qa_studies(seed))


def bench_sita_qa_studies_family(seed: int = _SEED + 4):
    """sita_qa_studies: synthetic correctness bench."""
    return _finite_blob(sita_qa_studies.bench_sita_qa_studies(seed))


def bench_vishnu_qa_studies_family(seed: int = _SEED + 5):
    """vishnu_qa_studies: synthetic correctness bench."""
    return _finite_blob(vishnu_qa_studies.bench_vishnu_qa_studies(seed))
