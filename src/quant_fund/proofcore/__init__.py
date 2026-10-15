"""PROOFCORE: proof-carrying, leakage-proof, statistically honest runtime.

Contract types load on first attribute access. Importing ``proofcore.cli``
must not pull the schema graph just to print help.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "CodeFingerprint",
    "DataAccessRecord",
    "DataManifestSummary",
    "EnvFingerprint",
    "LeakageFinding",
    "LeakageReport",
    "PitManifest",
    "PitManifestFile",
    "ProofBundleV1",
    "ProofcoreError",
    "RealityReport",
    "SignatureBlock",
    "TrialLedgerRow",
    "canonical_json_bytes",
    "merkle_root_hex",
    "sha256_hex_bytes",
]

_CONTRACTS = "quant_fund.proofcore.contracts"
_EXPORTS: dict[str, str] = {name: _CONTRACTS for name in __all__}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
