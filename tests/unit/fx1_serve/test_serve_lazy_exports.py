"""SYNTHETIC import-contract regressions for the lightweight serving facade."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PUBLIC_NAMES = [
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


def _run(script: str) -> None:
    source_root = Path(__file__).resolve().parents[3] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(source_root), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_import_does_not_load_serving_stack() -> None:
    _run(
        """
import sys
import fx1.serve
assert not any(name.startswith('fx1.serve.') for name in sys.modules)
assert 'quant_fund' not in sys.modules
"""
    )


def test_signing_exports_load_only_signing_dependencies() -> None:
    _run(
        """
import sys
from fx1.serve import build_manifest, sign_release, verify_release
from fx1.serve import signing
assert build_manifest is signing.build_manifest
assert sign_release is signing.sign_release
assert verify_release is signing.verify_release
assert 'fx1.serve.backends' not in sys.modules
assert 'fx1.serve.chat' not in sys.modules
assert 'fx1.serve.attestation' not in sys.modules
assert 'quant_fund' not in sys.modules
"""
    )


def test_public_names_and_discovery_are_preserved() -> None:
    import fx1.serve as serve

    assert serve.__all__ == _PUBLIC_NAMES
    assert set(serve._EXPORT_MODULES) == set(_PUBLIC_NAMES)
    assert set(_PUBLIC_NAMES).issubset(dir(serve))
    with pytest.raises(AttributeError, match="has no attribute 'not_an_export'"):
        getattr(serve, "not_an_export")


def test_exports_are_cached() -> None:
    _run(
        """
import fx1.serve as serve
first = serve.sign_release
assert vars(serve)['sign_release'] is first
assert serve.sign_release is first
"""
    )


def test_concurrent_first_access_has_stable_identity() -> None:
    _run(
        """
from concurrent.futures import ThreadPoolExecutor
import fx1.serve as serve
with ThreadPoolExecutor(max_workers=8) as pool:
    exports = list(pool.map(lambda _: serve.sign_release, range(64)))
assert all(export is exports[0] for export in exports)
"""
    )


def test_unknown_attribute_does_not_import_submodules() -> None:
    _run(
        """
import sys
import fx1.serve as serve
assert not hasattr(serve, 'not_an_export')
assert not any(name.startswith('fx1.serve.') for name in sys.modules)
"""
    )
