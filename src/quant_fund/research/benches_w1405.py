"""Wave-1405 bench adapters: audio-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    ambi_qa_studies,
    audio_qa_lite_studies,
    avsd_lite_studies,
    clotho_qa_studies,
    esc_qa_studies,
    music_avqa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14050


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ambi_qa_studies_family(seed: int = _SEED + 0):
    """ambi_qa_studies: synthetic correctness bench."""
    return _finite_blob(ambi_qa_studies.bench_ambi_qa_studies(seed))


def bench_audio_qa_lite_studies_family(seed: int = _SEED + 1):
    """audio_qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(audio_qa_lite_studies.bench_audio_qa_lite_studies(seed))


def bench_avsd_lite_studies_family(seed: int = _SEED + 2):
    """avsd_lite_studies: synthetic correctness bench."""
    return _finite_blob(avsd_lite_studies.bench_avsd_lite_studies(seed))


def bench_clotho_qa_studies_family(seed: int = _SEED + 3):
    """clotho_qa_studies: synthetic correctness bench."""
    return _finite_blob(clotho_qa_studies.bench_clotho_qa_studies(seed))


def bench_esc_qa_studies_family(seed: int = _SEED + 4):
    """esc_qa_studies: synthetic correctness bench."""
    return _finite_blob(esc_qa_studies.bench_esc_qa_studies(seed))


def bench_music_avqa_studies_family(seed: int = _SEED + 5):
    """music_avqa_studies: synthetic correctness bench."""
    return _finite_blob(music_avqa_studies.bench_music_avqa_studies(seed))
