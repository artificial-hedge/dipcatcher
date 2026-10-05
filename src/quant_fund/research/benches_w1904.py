"""Wave-1904 bench adapters: yokai-8 canon (SYNTHETIC only)."""

from quant_fund.models import (
    futakuchi_onna_qa_studies,
    ittan_momen_qa_studies,
    kasa_obake_qa_studies,
    mikoshi_nyudo_qa_studies,
    sunakake_babaa_qa_studies,
    umibozu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19040


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_futakuchi_onna_qa_studies_family(seed: int = _SEED + 0):
    """futakuchi_onna_qa_studies: synthetic correctness bench."""
    return _finite_blob(futakuchi_onna_qa_studies.bench_futakuchi_onna_qa_studies(seed))


def bench_ittan_momen_qa_studies_family(seed: int = _SEED + 1):
    """ittan_momen_qa_studies: synthetic correctness bench."""
    return _finite_blob(ittan_momen_qa_studies.bench_ittan_momen_qa_studies(seed))


def bench_kasa_obake_qa_studies_family(seed: int = _SEED + 2):
    """kasa_obake_qa_studies: synthetic correctness bench."""
    return _finite_blob(kasa_obake_qa_studies.bench_kasa_obake_qa_studies(seed))


def bench_mikoshi_nyudo_qa_studies_family(seed: int = _SEED + 3):
    """mikoshi_nyudo_qa_studies: synthetic correctness bench."""
    return _finite_blob(mikoshi_nyudo_qa_studies.bench_mikoshi_nyudo_qa_studies(seed))


def bench_sunakake_babaa_qa_studies_family(seed: int = _SEED + 4):
    """sunakake_babaa_qa_studies: synthetic correctness bench."""
    return _finite_blob(sunakake_babaa_qa_studies.bench_sunakake_babaa_qa_studies(seed))


def bench_umibozu_qa_studies_family(seed: int = _SEED + 5):
    """umibozu_qa_studies: synthetic correctness bench."""
    return _finite_blob(umibozu_qa_studies.bench_umibozu_qa_studies(seed))
