"""Wave-1792 bench adapters: sumerian-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    agga_qa_studies,
    babbar_qa_studies,
    enmerkar2_qa_studies,
    lugulbanda_qa_studies,
    ninsun_qa_studies,
    urukagina_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17920


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agga_qa_studies_family(seed: int = _SEED + 0):
    """agga_qa_studies: synthetic correctness bench."""
    return _finite_blob(agga_qa_studies.bench_agga_qa_studies(seed))


def bench_babbar_qa_studies_family(seed: int = _SEED + 1):
    """babbar_qa_studies: synthetic correctness bench."""
    return _finite_blob(babbar_qa_studies.bench_babbar_qa_studies(seed))


def bench_enmerkar2_qa_studies_family(seed: int = _SEED + 2):
    """enmerkar2_qa_studies: synthetic correctness bench."""
    return _finite_blob(enmerkar2_qa_studies.bench_enmerkar2_qa_studies(seed))


def bench_lugulbanda_qa_studies_family(seed: int = _SEED + 3):
    """lugulbanda_qa_studies: synthetic correctness bench."""
    return _finite_blob(lugulbanda_qa_studies.bench_lugulbanda_qa_studies(seed))


def bench_ninsun_qa_studies_family(seed: int = _SEED + 4):
    """ninsun_qa_studies: synthetic correctness bench."""
    return _finite_blob(ninsun_qa_studies.bench_ninsun_qa_studies(seed))


def bench_urukagina_qa_studies_family(seed: int = _SEED + 5):
    """urukagina_qa_studies: synthetic correctness bench."""
    return _finite_blob(urukagina_qa_studies.bench_urukagina_qa_studies(seed))
