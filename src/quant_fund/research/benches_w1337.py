"""Wave-1337 bench adapters: agentic-eval-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    android_env_studies,
    api_bank_studies,
    gaia_level_studies,
    video_game_studies,
    voyager_minecraft_studies,
    web_shopping_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13370


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_android_env_studies_family(seed: int = _SEED + 0):
    """android_env_studies: synthetic correctness bench."""
    return _finite_blob(android_env_studies.bench_android_env_studies(seed))


def bench_api_bank_studies_family(seed: int = _SEED + 1):
    """api_bank_studies: synthetic correctness bench."""
    return _finite_blob(api_bank_studies.bench_api_bank_studies(seed))


def bench_gaia_level_studies_family(seed: int = _SEED + 2):
    """gaia_level_studies: synthetic correctness bench."""
    return _finite_blob(gaia_level_studies.bench_gaia_level_studies(seed))


def bench_video_game_studies_family(seed: int = _SEED + 3):
    """video_game_studies: synthetic correctness bench."""
    return _finite_blob(video_game_studies.bench_video_game_studies(seed))


def bench_voyager_minecraft_studies_family(seed: int = _SEED + 4):
    """voyager_minecraft_studies: synthetic correctness bench."""
    return _finite_blob(voyager_minecraft_studies.bench_voyager_minecraft_studies(seed))


def bench_web_shopping_studies_family(seed: int = _SEED + 5):
    """web_shopping_studies: synthetic correctness bench."""
    return _finite_blob(web_shopping_studies.bench_web_shopping_studies(seed))
