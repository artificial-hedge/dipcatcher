"""Receipt verification facade — the only `research.receipt_v2` surface fx1 needs.

Pinned by the #2844 architecture-guard triage. fx1 and lower-layer
quant_fund modules must not import `quant_fund.research.receipt_v2`
directly; this module is the only sanctioned path to verify sealed
research evidence from outside the research layer.

The verify_* entry points are module-level call-time delegators: each one
imports `quant_fund.research.receipt_v2` inside the call and forwards to it,
so the upstream import stays inside function scope (the arch-guard's
sanctioned deferral mechanism — see ``configs/arch_boundaries.toml`` and
``scripts/check_import_boundaries.py``). This keeps the foundation layer
(``schemas``) free of upward edges into the research layer at import time.

Public surface (kept narrow on purpose):

- ``verify_receipt_file(path)``
- ``verify_receipt_payload(payload)``
- ``verify_receipt_bytes(data)``
- ``seal_receipt(receipt)`` — needed by ``registry.contract_probe`` to
  forge-then-reseal during deep coverage probes; this is a deliberate
  seam, not a leaky abstraction.

The full receipt model (``ReceiptV2``) and helpers (``build_receipt_v2``,
``wrap_receipt_v2``) stay in ``research.receipt_v2`` for in-research
callers only — fx1 never sees them, and the facade keeps the deeper
import chain (numpy, pydantic, the catalog/evalue_contracts/
impossible_fit/quantile_ladder chain) from leaking into the harness
foundation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Call-time delegators. The arch-guard treats module-scope imports of
# research-layer symbols as layer-order violations, so each deferrer below
# imports `quant_fund.research.receipt_v2` inside the call — the sanctioned
# function-scope deferral (see `configs/arch_boundaries.toml` and
# `scripts/check_import_boundaries.py`). These must be genuine delegators,
# NOT `...` stubs: PEP 562 module `__getattr__` only fires for *missing*
# attributes, so a module-level stub shadows it and silently returns None
# to every caller. Defaults mirror `research.receipt_v2` exactly so omitted
# arguments behave identically to the in-research call path.
def seal_receipt(receipt: Any) -> Any:  # type: ignore[no-untyped-def]
    from quant_fund.research import receipt_v2

    return receipt_v2.seal_receipt(receipt)


def verify_receipt_bytes(data: bytes, path: Any = Path("<memory>")) -> Any:  # type: ignore[no-untyped-def]
    from quant_fund.research import receipt_v2

    return receipt_v2.verify_receipt_bytes(data, path)


def verify_receipt_file(path: Any) -> Any:  # type: ignore[no-untyped-def]
    from quant_fund.research import receipt_v2

    return receipt_v2.verify_receipt_file(path)


def verify_receipt_payload(payload: Any, path: Any = Path("<memory>")) -> Any:  # type: ignore[no-untyped-def]
    from quant_fund.research import receipt_v2

    return receipt_v2.verify_receipt_payload(payload, path)


__all__ = [
    "seal_receipt",
    "verify_receipt_bytes",
    "verify_receipt_file",
    "verify_receipt_payload",
]


def __getattr__(name: str) -> Any:
    """Lazy attribute access — keeps the upstream import inside function scope.

    Per the arch-guard contract, function-scope imports are exempt from
    the layer-order check; module-scope imports are not. ``__getattr__``
    runs on first attribute access, so the upstream import happens at
    call time, not at module-import time.
    """
    if name in __all__:
        from quant_fund.research import receipt_v2

        return getattr(receipt_v2, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
