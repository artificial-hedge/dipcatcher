"""Credential redaction. The implementation lives on the harness.

:mod:`quant_fund.utils.secret_guard` owns the filter, the handler hook, and
the keystore mode checks. This module re-exports that surface so existing
``fx1.secret_guard`` imports, including the install performed by
:mod:`fx1`, keep working. The harness must not import :mod:`fx1` to scrub
logs.
"""

from __future__ import annotations

from quant_fund.utils.secret_guard import (
    SecretRedactionFilter,
    guard_keystore,
    install_secret_redaction,
    is_keystore_path,
    protect_keystore,
    redact_text,
    redacting_excepthook,
    refuse_world_readable,
    secrets_equal,
)

__all__ = [
    "SecretRedactionFilter",
    "guard_keystore",
    "install_secret_redaction",
    "is_keystore_path",
    "protect_keystore",
    "redact_text",
    "redacting_excepthook",
    "refuse_world_readable",
    "secrets_equal",
]
