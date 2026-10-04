"""Wave-1680 bench adapters: aztec-deity canon (SYNTHETIC only)."""

from quant_fund.models import (
    coatlicue_qa_studies,
    mictlan_qa_studies,
    mixcoatl_qa_studies,
    tlaloc_qa_studies,
    tonatiuh_qa_studies,
    xipe_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16800


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_coatlicue_qa_studies_family(seed: int = _SEED + 0):
    """coatlicue_qa_studies: synthetic correctness bench."""
    return _finite_blob(coatlicue_qa_studies.bench_coatlicue_qa_studies(seed))


def bench_mictlan_qa_studies_family(seed: int = _SEED + 1):
    """mictlan_qa_studies: synthetic correctness bench."""
    return _finite_blob(mictlan_qa_studies.bench_mictlan_qa_studies(seed))


def bench_mixcoatl_qa_studies_family(seed: int = _SEED + 2):
    """mixcoatl_qa_studies: synthetic correctness bench."""
    return _finite_blob(mixcoatl_qa_studies.bench_mixcoatl_qa_studies(seed))


def bench_tlaloc_qa_studies_family(seed: int = _SEED + 3):
    """tlaloc_qa_studies: synthetic correctness bench."""
    return _finite_blob(tlaloc_qa_studies.bench_tlaloc_qa_studies(seed))


def bench_tonatiuh_qa_studies_family(seed: int = _SEED + 4):
    """tonatiuh_qa_studies: synthetic correctness bench."""
    return _finite_blob(tonatiuh_qa_studies.bench_tonatiuh_qa_studies(seed))


def bench_xipe_qa_studies_family(seed: int = _SEED + 5):
    """xipe_qa_studies: synthetic correctness bench."""
    return _finite_blob(xipe_qa_studies.bench_xipe_qa_studies(seed))
