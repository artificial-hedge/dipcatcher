"""Vendor fetch scripts stay on the outbound guard."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from quant_fund.utils.outbound import OutboundRefused

_ROOT = Path(__file__).resolve().parents[3]
_PINNED_VERIFIERS = (
    "scripts/verify_auditor_bundle.py",
    "scripts/verify_corpus_proof.py",
    "scripts/verify_epoch_chain.py",
    "scripts/verify_ots_auditor.py",
)


def _vendor_http() -> Any:
    path = _ROOT / "scripts" / "vendor_http.py"
    spec = importlib.util.spec_from_file_location("vendor_http_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_scripts_do_not_call_urlopen() -> None:
    offenders = sorted(
        path.name
        for path in (_ROOT / "scripts").glob("fetch_*.py")
        if "urlopen" in path.read_text(encoding="utf-8")
    )
    assert offenders == []


def test_read_vendor_maps_a_refusal_to_urlerror(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _vendor_http()

    def refuse(*_args: object, **_kwargs: object) -> bytes:
        raise OutboundRefused("destination address is not allowed")

    monkeypatch.setattr(module, "read_http_redirects", refuse)
    request = urllib.request.Request("https://example.com/vendor")
    with pytest.raises(urllib.error.URLError, match="not allowed"):
        module.read_vendor(request, timeout=1)


def test_hash_pinned_verifiers_match_their_crown_jewel_bytes() -> None:
    pins = json.loads((_ROOT / "quality" / "crown_jewels.json").read_text(encoding="utf-8"))
    for relpath in _PINNED_VERIFIERS:
        digest = hashlib.sha256((_ROOT / relpath).read_bytes()).hexdigest()
        assert pins["files"][relpath] == digest
