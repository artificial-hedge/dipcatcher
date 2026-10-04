"""Wave-1741 bench adapters: hawaiian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    hina_qa_studies,
    kanaloa_qa_studies,
    kane_qa_studies,
    ku_qa_studies,
    lono_qa_studies,
    pele_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17410


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hina_qa_studies_family(seed: int = _SEED + 0):
    """hina_qa_studies: synthetic correctness bench."""
    return _finite_blob(hina_qa_studies.bench_hina_qa_studies(seed))


def bench_kanaloa_qa_studies_family(seed: int = _SEED + 1):
    """kanaloa_qa_studies: synthetic correctness bench."""
    return _finite_blob(kanaloa_qa_studies.bench_kanaloa_qa_studies(seed))


def bench_kane_qa_studies_family(seed: int = _SEED + 2):
    """kane_qa_studies: synthetic correctness bench."""
    return _finite_blob(kane_qa_studies.bench_kane_qa_studies(seed))


def bench_ku_qa_studies_family(seed: int = _SEED + 3):
    """ku_qa_studies: synthetic correctness bench."""
    return _finite_blob(ku_qa_studies.bench_ku_qa_studies(seed))


def bench_lono_qa_studies_family(seed: int = _SEED + 4):
    """lono_qa_studies: synthetic correctness bench."""
    return _finite_blob(lono_qa_studies.bench_lono_qa_studies(seed))


def bench_pele_qa_studies_family(seed: int = _SEED + 5):
    """pele_qa_studies: synthetic correctness bench."""
    return _finite_blob(pele_qa_studies.bench_pele_qa_studies(seed))
