"""fx-1 inference: backends, release signing, attestation ladder, cited chat.

Resolve the stable public exports on first use, so signing and attestation
utilities do not import the backend, chat, or research stack unnecessarily.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from fx1.serve.attestation import AttestationTier as AttestationTier
    from fx1.serve.attestation import OperatorProofManifest as OperatorProofManifest
    from fx1.serve.attestation import TEEQuote as TEEQuote
    from fx1.serve.attestation import attestation_ladder_status as attestation_ladder_status
    from fx1.serve.attestation import verify_quote as verify_quote
    from fx1.serve.backends import BackendNotConfiguredError as BackendNotConfiguredError
    from fx1.serve.backends import HostedK3Backend as HostedK3Backend
    from fx1.serve.backends import InferenceBackend as InferenceBackend
    from fx1.serve.backends import LocalFx1Backend as LocalFx1Backend
    from fx1.serve.backends import OpenAICompatBackend as OpenAICompatBackend
    from fx1.serve.backends import get_backend as get_backend
    from fx1.serve.chat import cited_complete as cited_complete
    from fx1.serve.signing import build_manifest as build_manifest
    from fx1.serve.signing import sign_release as sign_release
    from fx1.serve.signing import verify_release as verify_release

__all__ = [
    "AttestationTier",
    "BackendNotConfiguredError",
    "HostedK3Backend",
    "InferenceBackend",
    "LocalFx1Backend",
    "OpenAICompatBackend",
    "OperatorProofManifest",
    "TEEQuote",
    "attestation_ladder_status",
    "build_manifest",
    "cited_complete",
    "get_backend",
    "sign_release",
    "verify_quote",
    "verify_release",
]

_EXPORT_MODULES = {
    "AttestationTier": "attestation",
    "BackendNotConfiguredError": "backends",
    "HostedK3Backend": "backends",
    "InferenceBackend": "backends",
    "LocalFx1Backend": "backends",
    "OpenAICompatBackend": "backends",
    "OperatorProofManifest": "attestation",
    "TEEQuote": "attestation",
    "attestation_ladder_status": "attestation",
    "build_manifest": "signing",
    "cited_complete": "chat",
    "get_backend": "backends",
    "sign_release": "signing",
    "verify_quote": "attestation",
    "verify_release": "signing",
}


def __getattr__(name: str) -> object:
    """Cache a public export without eagerly importing unrelated modules."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = cast(object, getattr(module, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


def _install_module_entry_hook() -> None:
    """Arm the kill-file watcher only for ``python -m fx1.serve.api``.

    runpy sets ``sys.argv[0]`` to ``-m`` before this package is imported, and
    rewrites it to the module path only after ``loader.get_code`` returns.
    A normal import never sees that argv, so the hook stays out of tests.
    """
    import sys

    if sys.argv[:1] != ["-m"]:
        return
    from fx1.serve.module_entry import install_module_entry_hook

    install_module_entry_hook()


_install_module_entry_hook()
