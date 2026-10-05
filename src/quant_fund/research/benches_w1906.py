"""Wave-1906 bench adapters: yokai-10 canon (SYNTHETIC only)."""

from quant_fund.models import (
    amefuri_kozo_qa_studies,
    dosanjin_qa_studies,
    fuon_qa_studies,
    kejoro_qa_studies,
    nakisawame_qa_studies,
    yama_waro_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19060


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amefuri_kozo_qa_studies_family(seed: int = _SEED + 0):
    """amefuri_kozo_qa_studies: synthetic correctness bench."""
    return _finite_blob(amefuri_kozo_qa_studies.bench_amefuri_kozo_qa_studies(seed))


def bench_dosanjin_qa_studies_family(seed: int = _SEED + 1):
    """dosanjin_qa_studies: synthetic correctness bench."""
    return _finite_blob(dosanjin_qa_studies.bench_dosanjin_qa_studies(seed))


def bench_fuon_qa_studies_family(seed: int = _SEED + 2):
    """fuon_qa_studies: synthetic correctness bench."""
    return _finite_blob(fuon_qa_studies.bench_fuon_qa_studies(seed))


def bench_kejoro_qa_studies_family(seed: int = _SEED + 3):
    """kejoro_qa_studies: synthetic correctness bench."""
    return _finite_blob(kejoro_qa_studies.bench_kejoro_qa_studies(seed))


def bench_nakisawame_qa_studies_family(seed: int = _SEED + 4):
    """nakisawame_qa_studies: synthetic correctness bench."""
    return _finite_blob(nakisawame_qa_studies.bench_nakisawame_qa_studies(seed))


def bench_yama_waro_qa_studies_family(seed: int = _SEED + 5):
    """yama_waro_qa_studies: synthetic correctness bench."""
    return _finite_blob(yama_waro_qa_studies.bench_yama_waro_qa_studies(seed))
