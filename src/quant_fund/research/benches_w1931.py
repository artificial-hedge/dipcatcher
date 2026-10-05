"""Wave-1931 bench adapters: native-american-spirit canon (SYNTHETIC only)."""

from quant_fund.models import (
    deer_woman_qa_studies,
    kachina_qa_studies,
    manitou_qa_studies,
    mishipeshu_qa_studies,
    naagloshii_qa_studies,
    pukwudgie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19310


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_deer_woman_qa_studies_family(seed: int = _SEED + 0):
    """deer_woman_qa_studies: synthetic correctness bench."""
    return _finite_blob(deer_woman_qa_studies.bench_deer_woman_qa_studies(seed))



def bench_kachina_qa_studies_family(seed: int = _SEED + 1):
    """kachina_qa_studies: synthetic correctness bench."""
    return _finite_blob(kachina_qa_studies.bench_kachina_qa_studies(seed))



def bench_manitou_qa_studies_family(seed: int = _SEED + 2):
    """manitou_qa_studies: synthetic correctness bench."""
    return _finite_blob(manitou_qa_studies.bench_manitou_qa_studies(seed))



def bench_mishipeshu_qa_studies_family(seed: int = _SEED + 3):
    """mishipeshu_qa_studies: synthetic correctness bench."""
    return _finite_blob(mishipeshu_qa_studies.bench_mishipeshu_qa_studies(seed))



def bench_naagloshii_qa_studies_family(seed: int = _SEED + 4):
    """naagloshii_qa_studies: synthetic correctness bench."""
    return _finite_blob(naagloshii_qa_studies.bench_naagloshii_qa_studies(seed))



def bench_pukwudgie_qa_studies_family(seed: int = _SEED + 5):
    """pukwudgie_qa_studies: synthetic correctness bench."""
    return _finite_blob(pukwudgie_qa_studies.bench_pukwudgie_qa_studies(seed))
