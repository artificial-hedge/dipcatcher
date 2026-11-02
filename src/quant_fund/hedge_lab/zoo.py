"""Optional local Kronos zoo fetch. Never runs in CI. Disk budget gated."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from quant_fund.hedge_lab.resources import assert_disk_budget, lab_root
from quant_fund.models.robinhood_plus.constants import VARIANT_HUB
from quant_fund.utils.net_timeout import MAX_TRANSPORT_RETRIES

_HUB_TIMEOUT_S = 30.0
_HUB_RETRIES = 2


def _hub_retryable() -> tuple[type[BaseException], ...]:
    """Network faults worth one more snapshot attempt. Disk errors are not."""
    kinds: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        import requests
    except ImportError:
        return tuple(kinds)
    timeout = getattr(requests, "Timeout", None)
    if isinstance(timeout, type):
        kinds.append(timeout)
    return tuple(kinds)


def _invoke_snapshot(fn: Callable[..., str], kwargs: dict[str, Any]) -> str:
    """Pass ``etag_timeout`` only when this ``snapshot_download`` accepts it."""
    params = inspect.signature(fn).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in params}
    return fn(**accepted)


def _hub_snapshot(fn: Callable[..., str], **kwargs: Any) -> str:
    """One Hub snapshot with an explicit timeout and a hard retry cap."""
    if _HUB_RETRIES > MAX_TRANSPORT_RETRIES:
        raise RuntimeError("hub retries exceed the transport cap")
    call = {**kwargs, "etag_timeout": _HUB_TIMEOUT_S}
    retryable = _hub_retryable()
    last: BaseException | None = None
    for attempt in range(_HUB_RETRIES + 1):
        try:
            return _invoke_snapshot(fn, call)
        except retryable as exc:
            last = exc
            if attempt >= _HUB_RETRIES:
                raise
    raise RuntimeError("hub snapshot retries exhausted") from last


def fetch_kronos_variant(variant: str = "small") -> dict[str, Any]:
    """Snapshot one Kronos Hub pair into ``third_party/kronos_weights``.

    Requires ``huggingface_hub``. Does not change ``blend_weight``. Missing
    extras stamp ``skipped``.
    """
    if variant not in VARIANT_HUB:
        return {"status": "unknown_variant", "variant": variant}
    spec = VARIANT_HUB[variant]
    dest_root = lab_root() / "third_party" / "kronos_weights"
    dest_root.mkdir(parents=True, exist_ok=True)
    assert_disk_budget(extra_bytes=2 * 1024**3)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {
            "status": "skipped_no_huggingface_hub",
            "variant": variant,
            "research_only": True,
        }
    tokenizer = str(spec["tokenizer"])
    model = str(spec["model"])
    tok_dir = dest_root / tokenizer.rsplit("/", 1)[-1]
    model_dir = dest_root / model.rsplit("/", 1)[-1]
    tok_path = _hub_snapshot(
        snapshot_download,
        repo_id=tokenizer,
        revision=str(spec["tokenizer_rev"]),
        local_dir=str(tok_dir),
    )
    model_path = _hub_snapshot(
        snapshot_download,
        repo_id=model,
        revision=str(spec["model_rev"]),
        local_dir=str(model_dir),
    )
    return {
        "status": "ok",
        "variant": variant,
        "tokenizer": tokenizer,
        "model": model,
        "tokenizer_path": str(tok_path),
        "model_path": str(model_path),
        "research_only": True,
        "blend_weight": 0.0,
        "sizes_book": False,
    }
