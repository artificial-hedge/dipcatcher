"""dispatch_audit — pin the native-backend dispatch contract.

``QUANT_FUND_NATIVE`` decides which engine the whole package runs on, and
its edges are load-bearing: ``rust`` must *fail closed* (ImportError with
remediation) when the extension is missing — a silent fallback would let
a production deployment run the slow backend unknowingly. ``python`` must
force the reference path even when rust is importable. The flag is read
once at import — reload-time drift is the audit surface.

Sealed ``dispatch_audit.v1``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["dispatch_audit", "dispatch_audit_bench"]

_ENV = "QUANT_FUND_NATIVE"


def _flag_outcome(value: str) -> str:
    """Probe ``_flag`` directly for parse outcomes."""
    import quant_fund.native as native

    saved = os.environ.get(_ENV)
    try:
        os.environ[_ENV] = value
        try:
            return native._flag()  # noqa: SLF001 — auditing the private parser is the point
        except ValueError:
            return "raise:ValueError"
    finally:
        if saved is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = saved


def _reload_backend(value: str) -> str:
    """Reload the module under an env value and report its BACKEND."""
    import quant_fund.native as native

    saved = os.environ.get(_ENV)
    try:
        os.environ[_ENV] = value
        try:
            importlib.reload(native)
            return str(native.BACKEND)
        except ImportError:
            return "raise:ImportError"
    finally:
        if saved is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = saved
        importlib.reload(native)


def dispatch_audit() -> dict[str, Any]:
    results: dict[str, Any] = {}

    results["flag_parsing"] = {
        "auto": _flag_outcome("auto"),
        "empty": _flag_outcome(""),
        "whitespace": _flag_outcome("   "),
        "python_alias": _flag_outcome("PYTHON "),
        "numpy_alias": _flag_outcome("numpy"),
        "rust_alias": _flag_outcome("Native"),
        "garbage": _flag_outcome("rusty"),
    }

    results["backend_routing"] = {
        "python_forces_python": _reload_backend("python") == "python",
        "auto_backend": _reload_backend("auto"),
    }

    # rust-forced semantics depend on whether the extension exists in this
    # environment — pin whichever contract is live, and assert it's the
    # honest one either way.
    import quant_fund.native as native

    rust_available = native._load_rust("auto") is not None  # noqa: SLF001
    if rust_available:
        results["rust_forced"] = {
            "backend": _reload_backend("rust"),
            "ok": _reload_backend("rust") == "rust",
        }
    else:
        results["rust_forced"] = {
            "backend": _reload_backend("rust"),
            "ok": _reload_backend("rust") == "raise:ImportError",
            "note": "extension absent — rust-forced must fail closed, not fall back",
        }

    import numpy as np

    x = np.arange(30, dtype=float)
    os.environ[_ENV] = "python"
    importlib.reload(native)
    mean_py = native.rolling_mean(x, 5)
    os.environ[_ENV] = "auto"
    importlib.reload(native)
    mean_auto = native.rolling_mean(x, 5)
    results["parity_across_dispatch"] = {
        "ok": bool(np.allclose(mean_py, mean_auto, atol=1e-12, equal_nan=True)),
        "note": "python-forced and auto dispatch compute identical numbers",
    }
    return results


def dispatch_audit_bench() -> dict[str, Any]:
    r = dispatch_audit()
    fp = r["flag_parsing"]
    ok = (
        fp["auto"] == "auto"
        and fp["empty"] == "auto"
        and fp["whitespace"] == "auto"
        and fp["python_alias"] == "python"
        and fp["numpy_alias"] == "python"
        and fp["rust_alias"] == "rust"
        and fp["garbage"] == "raise:ValueError"
        and r["backend_routing"]["python_forces_python"]
        and r["rust_forced"]["ok"]
        and r["parity_across_dispatch"]["ok"]
    )
    payload: dict[str, Any] = {
        "kind": "dispatch_audit",
        "schema": "dispatch_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Dispatch contract pinned: flag aliases/whitespace/garbage, "
            "python-forced override, rust-forced fail-closed when the "
            "extension is absent, and cross-backend numerical parity."
            if ok
            else f"DISPATCH DRIFT: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
