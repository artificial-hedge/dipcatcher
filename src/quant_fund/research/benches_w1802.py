"""Wave-1802 bench adapters: maori-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    awhi2_qa_studies,
    hine3_qa_studies,
    kaikoura_qa_studies,
    moana2_qa_studies,
    ranginui2_qa_studies,
    tanemahuta2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18020


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_awhi2_qa_studies_family(seed: int = _SEED + 0):
    """awhi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(awhi2_qa_studies.bench_awhi2_qa_studies(seed))


def bench_hine3_qa_studies_family(seed: int = _SEED + 1):
    """hine3_qa_studies: synthetic correctness bench."""
    return _finite_blob(hine3_qa_studies.bench_hine3_qa_studies(seed))


def bench_kaikoura_qa_studies_family(seed: int = _SEED + 2):
    """kaikoura_qa_studies: synthetic correctness bench."""
    return _finite_blob(kaikoura_qa_studies.bench_kaikoura_qa_studies(seed))


def bench_moana2_qa_studies_family(seed: int = _SEED + 3):
    """moana2_qa_studies: synthetic correctness bench."""
    return _finite_blob(moana2_qa_studies.bench_moana2_qa_studies(seed))


def bench_ranginui2_qa_studies_family(seed: int = _SEED + 4):
    """ranginui2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ranginui2_qa_studies.bench_ranginui2_qa_studies(seed))


def bench_tanemahuta2_qa_studies_family(seed: int = _SEED + 5):
    """tanemahuta2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanemahuta2_qa_studies.bench_tanemahuta2_qa_studies(seed))
