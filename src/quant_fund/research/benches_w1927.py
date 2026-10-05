"""Wave-1927 bench adapters: inuit-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    amautalik_qa_studies,
    ijiraq_qa_studies,
    mahaha_qa_studies,
    qivittoq_qa_studies,
    tornit_qa_studies,
    tupilaq_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19270


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amautalik_qa_studies_family(seed: int = _SEED + 0):
    """amautalik_qa_studies: synthetic correctness bench."""
    return _finite_blob(amautalik_qa_studies.bench_amautalik_qa_studies(seed))


def bench_ijiraq_qa_studies_family(seed: int = _SEED + 1):
    """ijiraq_qa_studies: synthetic correctness bench."""
    return _finite_blob(ijiraq_qa_studies.bench_ijiraq_qa_studies(seed))


def bench_mahaha_qa_studies_family(seed: int = _SEED + 2):
    """mahaha_qa_studies: synthetic correctness bench."""
    return _finite_blob(mahaha_qa_studies.bench_mahaha_qa_studies(seed))


def bench_qivittoq_qa_studies_family(seed: int = _SEED + 3):
    """qivittoq_qa_studies: synthetic correctness bench."""
    return _finite_blob(qivittoq_qa_studies.bench_qivittoq_qa_studies(seed))


def bench_tornit_qa_studies_family(seed: int = _SEED + 4):
    """tornit_qa_studies: synthetic correctness bench."""
    return _finite_blob(tornit_qa_studies.bench_tornit_qa_studies(seed))


def bench_tupilaq_qa_studies_family(seed: int = _SEED + 5):
    """tupilaq_qa_studies: synthetic correctness bench."""
    return _finite_blob(tupilaq_qa_studies.bench_tupilaq_qa_studies(seed))
