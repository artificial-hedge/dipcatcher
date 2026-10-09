"""Research-side entry point for the ``receipt.v2`` envelope (#2991).

The envelope contract and its verifier now live in
:mod:`quant_fund.schemas.receipt_v2_leaf` — a dependency-safe leaf with **no
module-scope** import of the research layer. fx1, the microstructure map
modules, and the registry contract probe import that leaf directly, which is
what lets the ``fx1-harness-surface`` allowlist drop its
``quant_fund.research.receipt_v2`` entry (d1082fd9 had widened it to silence
eight harness-surface sites) and lets five layer-order baselines be deleted.

This module stays as the research-side entry point so that every existing
caller is unchanged, and so the published ``receipt_v2.schema.json`` keeps
sitting beside the module research lanes import.

Two things intentionally live here rather than in the leaf:

* research-lane *writers* that need research policy at import time;
* the schema JSON resource, which is addressed relative to this package.

The public surface is re-exported explicitly so type checkers and linters see
it; ``__getattr__`` forwards anything else (tests reach for a few private lane
tables) so this shim cannot quietly drift out of sync with the leaf.
"""

from __future__ import annotations

from typing import Any

from quant_fund.schemas import receipt_v2_leaf as _leaf
from quant_fund.schemas.receipt_v2_leaf import (  # noqa: F401
    RECEIPT_V2_SCHEMA,
    RECEIPT_V2_SCHEMA_FILE,
    RECEIPT_V2_SCHEMA_VERSION,
    RECEIPT_V2_VERDICTS,
    REQUIRED_ENV_PACKAGES,
    BlasDependency,
    ReceiptEnvironment,
    ReceiptV2,
    ReceiptVerification,
    ThreadpoolInfo,
    build_receipt_v2,
    code_fingerprint,
    environment_fingerprint,
    receipt_v2_json_schema,
    seal_receipt,
    verify_receipt_bytes,
    verify_receipt_file,
    verify_receipt_payload,
    wrap_receipt_v2,
)

__all__ = [
    "RECEIPT_V2_SCHEMA",
    "RECEIPT_V2_SCHEMA_FILE",
    "RECEIPT_V2_SCHEMA_VERSION",
    "RECEIPT_V2_VERDICTS",
    "REQUIRED_ENV_PACKAGES",
    "BlasDependency",
    "ReceiptEnvironment",
    "ReceiptV2",
    "ReceiptVerification",
    "ThreadpoolInfo",
    "build_receipt_v2",
    "code_fingerprint",
    "environment_fingerprint",
    "receipt_v2_json_schema",
    "seal_receipt",
    "verify_receipt_bytes",
    "verify_receipt_file",
    "verify_receipt_payload",
    "wrap_receipt_v2",
]


def __getattr__(name: str) -> Any:
    """Forward any remaining attribute to the leaf, private helpers included."""
    try:
        return getattr(_leaf, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; the receipt.v2 leaf "
            f"{_leaf.__name__!r} does not define it either"
        ) from None


def __dir__() -> list[str]:
    return sorted({*__all__, *dir(_leaf)})
