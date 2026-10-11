from quant_fund.utils.atomicio import (
    atomic_write_bytes,
    atomic_write_parquet,
    atomic_write_text,
    publish_text_once,
)
from quant_fund.utils.hashing import fingerprint, hash_bytes, hash_file
from quant_fund.utils.logging import configure_logging, get_logger
from quant_fund.utils.seeds import set_global_seed

# ``quant_fund.utils.numeric`` imports numpy at module scope; importing it
# here would force every consumer of ``quant_fund.utils`` to pay the ~1.5s
# numpy cold-start. The two functions are tiny and are re-exported via
# __getattr__ so a downstream ``from quant_fund.utils import clip_positive``
# still works without bringing numpy in at import time.

__all__ = [
    "atomic_write_bytes",
    "atomic_write_parquet",
    "atomic_write_text",
    "publish_text_once",
    "clip_positive",
    "configure_logging",
    "fingerprint",
    "get_logger",
    "hash_bytes",
    "hash_file",
    "require_finite",
    "set_global_seed",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in ("clip_positive", "require_finite"):
        from quant_fund.utils import numeric

        return getattr(numeric, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().get("__all__", [])) | set(__all__))
