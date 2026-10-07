"""Receipt verification facade — the only `research.receipt_v2` surface fx1 needs.

Pinned by the #2844 architecture-guard triage. fx1 and lower-layer
quant_fund modules must not import `quant_fund.research.receipt_v2`
directly; this module is the only sanctioned path to verify sealed
research evidence from outside the research layer.

The verify_* entry points are re-exported via module-level ``__getattr__``
so the `quant_fund.research.receipt_v2` import is lazy and stays inside
function scope (the arch-guard's sanctioned deferral mechanism — see
``configs/arch_boundaries.toml`` and ``scripts/check_import_boundaries.py``).
This keeps the foundation layer (``schemas``) free of upward edges into
the research layer.

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


# Delegating facades — the bodies import ``research.receipt_v2`` lazily at
# call time (function scope), which the arch-guard exempts from the
# layer-order check. These MUST be real delegations: bare ``...`` stubs
# permanently shadow ``__getattr__`` below and silently return ``None`` to
# every caller.
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
