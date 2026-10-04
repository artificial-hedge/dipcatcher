"""Wave-1737 bench adapters: aztec-deity-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    coatlicue_qa_studies,
    coyolxauhqui_qa_studies,
    huitzilopochtli_qa_studies,
    mictlantecuhtli_qa_studies,
    tlaloc_qa_studies,
    tonatiuh_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17370


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


def bench_coyolxauhqui_qa_studies_family(seed: int = _SEED + 1):
    """coyolxauhqui_qa_studies: synthetic correctness bench."""
    return _finite_blob(coyolxauhqui_qa_studies.bench_coyolxauhqui_qa_studies(seed))


def bench_huitzilopochtli_qa_studies_family(seed: int = _SEED + 2):
    """huitzilopochtli_qa_studies: synthetic correctness bench."""
    return _finite_blob(huitzilopochtli_qa_studies.bench_huitzilopochtli_qa_studies(seed))


def bench_mictlantecuhtli_qa_studies_family(seed: int = _SEED + 3):
    """mictlantecuhtli_qa_studies: synthetic correctness bench."""
    return _finite_blob(mictlantecuhtli_qa_studies.bench_mictlantecuhtli_qa_studies(seed))


def bench_tlaloc_qa_studies_family(seed: int = _SEED + 4):
    """tlaloc_qa_studies: synthetic correctness bench."""
    return _finite_blob(tlaloc_qa_studies.bench_tlaloc_qa_studies(seed))


def bench_tonatiuh_qa_studies_family(seed: int = _SEED + 5):
    """tonatiuh_qa_studies: synthetic correctness bench."""
    return _finite_blob(tonatiuh_qa_studies.bench_tonatiuh_qa_studies(seed))
