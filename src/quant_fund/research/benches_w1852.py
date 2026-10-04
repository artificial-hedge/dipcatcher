"""Wave-1852 bench adapters: numidian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    aulisua_qa_studies,
    gurzil_qa_studies,
    iguc_qa_studies,
    lallus_qa_studies,
    macurgum_qa_studies,
    melyakina_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18520


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aulisua_qa_studies_family(seed: int = _SEED + 0):
    """aulisua_qa_studies: synthetic correctness bench."""
    return _finite_blob(aulisua_qa_studies.bench_aulisua_qa_studies(seed))


def bench_gurzil_qa_studies_family(seed: int = _SEED + 1):
    """gurzil_qa_studies: synthetic correctness bench."""
    return _finite_blob(gurzil_qa_studies.bench_gurzil_qa_studies(seed))


def bench_iguc_qa_studies_family(seed: int = _SEED + 2):
    """iguc_qa_studies: synthetic correctness bench."""
    return _finite_blob(iguc_qa_studies.bench_iguc_qa_studies(seed))


def bench_lallus_qa_studies_family(seed: int = _SEED + 3):
    """lallus_qa_studies: synthetic correctness bench."""
    return _finite_blob(lallus_qa_studies.bench_lallus_qa_studies(seed))


def bench_macurgum_qa_studies_family(seed: int = _SEED + 4):
    """macurgum_qa_studies: synthetic correctness bench."""
    return _finite_blob(macurgum_qa_studies.bench_macurgum_qa_studies(seed))


def bench_melyakina_qa_studies_family(seed: int = _SEED + 5):
    """melyakina_qa_studies: synthetic correctness bench."""
    return _finite_blob(melyakina_qa_studies.bench_melyakina_qa_studies(seed))
