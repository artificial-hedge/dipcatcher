"""Wave-1900 bench adapters: yokai-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    enenra_qa_studies,
    goryo_qa_studies,
    kiyohime_qa_studies,
    kodama_shirakawa_qa_studies,
    nure_onna_qa_studies,
    yurei_muzen_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19000


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_enenra_qa_studies_family(seed: int = _SEED + 0):
    """enenra_qa_studies: synthetic correctness bench."""
    return _finite_blob(enenra_qa_studies.bench_enenra_qa_studies(seed))


def bench_goryo_qa_studies_family(seed: int = _SEED + 1):
    """goryo_qa_studies: synthetic correctness bench."""
    return _finite_blob(goryo_qa_studies.bench_goryo_qa_studies(seed))


def bench_kiyohime_qa_studies_family(seed: int = _SEED + 2):
    """kiyohime_qa_studies: synthetic correctness bench."""
    return _finite_blob(kiyohime_qa_studies.bench_kiyohime_qa_studies(seed))


def bench_kodama_shirakawa_qa_studies_family(seed: int = _SEED + 3):
    """kodama_shirakawa_qa_studies: synthetic correctness bench."""
    return _finite_blob(kodama_shirakawa_qa_studies.bench_kodama_shirakawa_qa_studies(seed))


def bench_nure_onna_qa_studies_family(seed: int = _SEED + 4):
    """nure_onna_qa_studies: synthetic correctness bench."""
    return _finite_blob(nure_onna_qa_studies.bench_nure_onna_qa_studies(seed))


def bench_yurei_muzen_qa_studies_family(seed: int = _SEED + 5):
    """yurei_muzen_qa_studies: synthetic correctness bench."""
    return _finite_blob(yurei_muzen_qa_studies.bench_yurei_muzen_qa_studies(seed))
