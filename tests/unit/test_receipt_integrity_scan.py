"""Tests for ``scripts/receipt_integrity_scan.py``.

The scanner is a *digest* check, not a full verifier. These tests pin the
two properties that matter:

1. It is **fail-closed** — a receipt whose ``receipt_sha256`` does not
   describe its content is a failure, not a warning.
2. It does **not** manufacture failures — a schema that carries its own seal
   (no ``receipt_sha256``) is reported as ``no_selfseal``, not as a mismatch.
   The 20-minute ``make receipts-reverify`` gate mixed those two together;
   separating them is the point of this tool.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCAN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "receipt_integrity_scan.py"
_spec = importlib.util.spec_from_file_location("receipt_integrity_scan", _SCAN_PATH)
assert _spec and _spec.loader, "receipt_integrity_scan.py must be importable"
scan_mod = importlib.util.module_from_spec(_spec)
sys.modules["receipt_integrity_scan"] = scan_mod
_spec.loader.exec_module(scan_mod)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes  # noqa: E402


def _sealed(**fields: Any) -> dict[str, Any]:
    """Build a receipt whose self-seal is correct by construction."""
    payload: dict[str, Any] = dict(fields)
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


def _write(directory: Path, name: str, payload: Any) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_matching_self_seal_is_ok(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed(schema="x.v1", kind="k", value=1))
    assert scan_mod.classify(tmp_path / "a.json")[0] == "ok"


def test_tampered_body_fails_closed(tmp_path: Path) -> None:
    """Editing a sealed field must break the seal — the core contract."""
    path = _write(tmp_path, "a.json", _sealed(schema="x.v1", kind="k", value=1))
    payload = json.loads(path.read_text())
    payload["value"] = 999  # silent edit, seal untouched
    path.write_text(json.dumps(payload), encoding="utf-8")
    status, _ = scan_mod.classify(path)
    assert status == "mismatch"


def test_missing_selfseal_is_not_a_failure(tmp_path: Path) -> None:
    """A different schema's seal must not be reported as corruption."""
    _write(tmp_path, "b.json", {"schema": "forward_record_preregistration.v1", "_seal": {}})
    assert scan_mod.classify(tmp_path / "b.json")[0] == "no_selfseal"


def test_alternate_ascii_escaping_convention_is_intact(tmp_path: Path) -> None:
    """Receipts sealed with ``ensure_ascii=True`` are INTACT, not corrupt.

    ``canonical_json_bytes`` serializes with ``ensure_ascii=False``, so a payload
    containing an em dash or arrow produces different bytes than
    ``json.dumps``'s default ASCII-escaping form. Two committed receipts are
    sealed that way. Flagging them as corruption would be the same
    false-positive conflation this scanner exists to prevent, so they get their
    own non-failing status.
    """
    payload: dict[str, Any] = {"schema": "x.v1", "kind": "k", "note": "queued—running → done"}
    escaped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert "\\u2014" in escaped, "fixture must actually contain non-ASCII to be meaningful"
    payload["receipt_sha256"] = hash_bytes(escaped.encode("utf-8"))

    path = _write(tmp_path, "alt.json", payload)
    status, detail = scan_mod.classify(path)
    assert status == "ok_alt_convention"
    assert "ensure_ascii=True" in detail

    # and crucially: it must not be a failure
    report = scan_mod.scan(tmp_path)
    assert report["counts"]["mismatch"] == 0
    assert report["verdict"] == "PASS"


def test_genuine_drift_still_fails_under_both_conventions(tmp_path: Path) -> None:
    """A real content edit must fail no matter which convention sealed it."""
    for label, kwargs in (
        ("canonical", None),
        ("ascii", {"ensure_ascii": True}),
    ):
        payload: dict[str, Any] = {"schema": "x.v1", "kind": "k", "v": 1}
        if kwargs is None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        else:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":"), **kwargs)
        payload["receipt_sha256"] = hash_bytes(body.encode("utf-8"))
        path = _write(tmp_path, f"{label}.json", payload)
        edited = json.loads(path.read_text())
        edited["v"] = 2
        path.write_text(json.dumps(edited), encoding="utf-8")
        assert scan_mod.classify(path)[0] == "mismatch", label


def test_sidecar_seals_are_not_collected_as_receipts(tmp_path: Path) -> None:
    """``*.seal.json`` files are seal evidence, not receipts (see ci.receipt_paths)."""
    _write(tmp_path, "r.json", _sealed(schema="x.v1", kind="k"))
    _write(tmp_path, "r.json.seal.json", {"schema": "prereg_seal.v1", "sha256": "0" * 64})
    report = scan_mod.scan(tmp_path)
    assert report["scanned"] == 1


def test_scan_verdict_and_exit_codes(tmp_path: Path) -> None:
    _write(tmp_path, "good.json", _sealed(schema="x.v1", kind="k"))
    assert scan_mod.scan(tmp_path)["verdict"] == "PASS"
    assert scan_mod.main([str(tmp_path), "--json"]) == 0

    bad = _write(tmp_path, "bad.json", _sealed(schema="y.v1", kind="k"))
    payload = json.loads(bad.read_text())
    payload["kind"] = "tampered"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    report = scan_mod.scan(tmp_path)
    assert report["verdict"] == "FAIL"
    assert report["counts"]["mismatch"] == 1
    assert scan_mod.main([str(tmp_path), "--json"]) == 1


def test_unparseable_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert scan_mod.scan(tmp_path)["verdict"] == "FAIL"


def test_empty_directory_does_not_pass_silently(tmp_path: Path) -> None:
    """No receipts scanned must not read as a clean bill of health."""
    report = scan_mod.scan(tmp_path)
    assert report["scanned"] == 0
    assert report["verdict"] in {"PASS", "FAIL"}


@pytest.mark.parametrize("name", ["receipts"])
def test_repo_receipts_scan_is_reproducible(name: str) -> None:
    """Scanning the real receipts dir must be deterministic."""
    root = Path(__file__).resolve().parents[2] / name
    if not root.is_dir():
        pytest.skip(f"{name}/ not present")
    first = scan_mod.scan(root)
    second = scan_mod.scan(root)
    assert first["counts"] == second["counts"]
    assert first["verdict"] == second["verdict"]
