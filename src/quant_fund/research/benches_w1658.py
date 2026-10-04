"""Wave-1658 bench adapters: mesoamerican-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    ahuizotl_qa_studies,
    alicanto_qa_studies,
    cadejo_qa_studies,
    cipactli_qa_studies,
    jinn_qa_studies,
    quetzalcoat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16580


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ahuizotl_qa_studies_family(seed: int = _SEED + 0):
    """ahuizotl_qa_studies: synthetic correctness bench."""
    return _finite_blob(ahuizotl_qa_studies.bench_ahuizotl_qa_studies(seed))


def bench_alicanto_qa_studies_family(seed: int = _SEED + 1):
    """alicanto_qa_studies: synthetic correctness bench."""
    return _finite_blob(alicanto_qa_studies.bench_alicanto_qa_studies(seed))


def bench_cadejo_qa_studies_family(seed: int = _SEED + 2):
    """cadejo_qa_studies: synthetic correctness bench."""
    return _finite_blob(cadejo_qa_studies.bench_cadejo_qa_studies(seed))


def bench_cipactli_qa_studies_family(seed: int = _SEED + 3):
    """cipactli_qa_studies: synthetic correctness bench."""
    return _finite_blob(cipactli_qa_studies.bench_cipactli_qa_studies(seed))


def bench_jinn_qa_studies_family(seed: int = _SEED + 4):
    """jinn_qa_studies: synthetic correctness bench."""
    return _finite_blob(jinn_qa_studies.bench_jinn_qa_studies(seed))


def bench_quetzalcoat_qa_studies_family(seed: int = _SEED + 5):
    """quetzalcoat_qa_studies: synthetic correctness bench."""
    return _finite_blob(quetzalcoat_qa_studies.bench_quetzalcoat_qa_studies(seed))
