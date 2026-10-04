"""Wave-1648 bench adapters: slavic-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    aitvaras_qa_studies,
    bilwis_qa_studies,
    indus_qa_studies,
    kudlak_qa_studies,
    viy_qa_studies,
    zilant_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16480


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aitvaras_qa_studies_family(seed: int = _SEED + 0):
    """aitvaras_qa_studies: synthetic correctness bench."""
    return _finite_blob(aitvaras_qa_studies.bench_aitvaras_qa_studies(seed))


def bench_bilwis_qa_studies_family(seed: int = _SEED + 1):
    """bilwis_qa_studies: synthetic correctness bench."""
    return _finite_blob(bilwis_qa_studies.bench_bilwis_qa_studies(seed))


def bench_indus_qa_studies_family(seed: int = _SEED + 2):
    """indus_qa_studies: synthetic correctness bench."""
    return _finite_blob(indus_qa_studies.bench_indus_qa_studies(seed))


def bench_kudlak_qa_studies_family(seed: int = _SEED + 3):
    """kudlak_qa_studies: synthetic correctness bench."""
    return _finite_blob(kudlak_qa_studies.bench_kudlak_qa_studies(seed))


def bench_viy_qa_studies_family(seed: int = _SEED + 4):
    """viy_qa_studies: synthetic correctness bench."""
    return _finite_blob(viy_qa_studies.bench_viy_qa_studies(seed))


def bench_zilant_qa_studies_family(seed: int = _SEED + 5):
    """zilant_qa_studies: synthetic correctness bench."""
    return _finite_blob(zilant_qa_studies.bench_zilant_qa_studies(seed))
