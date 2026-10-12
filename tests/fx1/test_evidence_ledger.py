"""Evidence ledger: receipt verification, citations, and composed answers."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import fx1.interactive.evidence_ledger as evidence_module
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive.evidence_ledger import (
    CitationProblem,
    EvidenceItem,
    EvidenceKind,
    EvidenceLedger,
    Verdict,
    check_citations,
    compose_answer,
    extract_data_label,
    extract_receipt_paths,
    verify_receipt,
)

HASH = "ab" * 32
OTHER_HASH = "cd" * 32


@pytest.fixture(autouse=True)
def _fx1_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ok_verifier(schema: str = "demo.v1") -> Callable[[Path], Mapping[str, Any]]:
    def verify(path: Path) -> Mapping[str, Any]:
        del path
        return {"valid": True, "schema": schema}

    return verify


def _plant(tmp_path: Path, data: bytes, name: str = "serial_watch.json") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


@pytest.fixture(scope="module")
def sealed_receipt_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """A real serial_watch receipt sealed by the lab writer."""
    from quant_fund.research.serial_watch import serial_report, write_serial_receipt

    directory = tmp_path_factory.mktemp("sealed-serial")
    receipt = serial_report(
        [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.25, 0.75],
        n_lags=1,
        data_label="SYNTHETIC",
    )
    path = write_serial_receipt(receipt, directory)
    return path.read_bytes()


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("receipt=a.json\n", ["a.json"]),
        ("receipt: b.json\n", ["b.json"]),
        ('receipt="c.json"\n', ["c.json"]),
        ("receipt='d.json'\n", ["d.json"]),
        ("wrote receipt=e.json\n", ["e.json"]),
        ("receipt=\n", []),
        ("receipt=../x.json\n", []),
        ("receipt=foo/../../x\n", []),
        ("receipt=a..b.json\n", []),
        ("receipt=ab\x00c.json\n", []),
        ("receipt=a.json\nreceipt=a.json\n", ["a.json"]),
        ("receipt=a.json\nreceipt=b.json\n", ["a.json", "b.json"]),
        ("Receipt=a.json\n", []),
        ("prereceipt=a.json\n", []),
        ("receipt:   spaced.json\n", ["spaced.json"]),
        ("no marker\n", []),
        ("receipt=a.json\r\nreceipt=b.json\r\n", ["a.json", "b.json"]),
        ("receipt=''\n", []),
    ],
)
def test_extract_receipt_paths(stdout: str, expected: list[str]) -> None:
    assert extract_receipt_paths(stdout) == expected


def test_extract_receipt_paths_caps_at_twenty() -> None:
    lines = [f"receipt=p{index}.json" for index in range(25)]
    assert extract_receipt_paths("\n".join(lines)) == [f"p{index}.json" for index in range(20)]


def test_extract_receipt_paths_rejects_overlong_values() -> None:
    ok = "a" * 507 + ".json"
    assert len(ok) == 512
    assert extract_receipt_paths(f"receipt={ok}") == [ok]
    assert extract_receipt_paths(f"receipt={ok}x") == []


def test_extract_receipt_paths_honors_custom_marker() -> None:
    stdout = "out=a.json\nreceipt=b.json\nout: c.json\n"
    assert extract_receipt_paths(stdout, marker="out=") == ["a.json", "c.json"]


def test_extract_receipt_paths_does_not_touch_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("filesystem")

    monkeypatch.setattr(Path, "exists", boom)
    assert extract_receipt_paths("receipt=does-not-exist.json") == ["does-not-exist.json"]


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("", "UNKNOWN"),
        ("no label\n", "UNKNOWN"),
        ("the file is SYNTHETIC\n", "UNKNOWN"),
        ("SYNTHETIC\n", "SYNTHETIC"),
        ("DATA_LABEL=REAL\n", "REAL"),
        ("DATA_LABEL=synthetic\n", "SYNTHETIC"),
        ("DATA_LABEL=USER_SUPPLIED\n", "USER_SUPPLIED"),
        ("DATA_LABEL=CORPUS\n", "CORPUS"),
        ("DATA_LABEL=\n", "UNKNOWN"),
        ("  DATA_LABEL = REAL  \n", "REAL"),
        ('DATA_LABEL="REAL"\n', "REAL"),
        ("DATA_LABEL=REAL\nSYNTHETIC\n", "SYNTHETIC"),
        ("DATA_LABEL=SYNTHETIC\nDATA_LABEL=REAL\n", "SYNTHETIC"),
        ("DATA_LABEL=REAL\nDATA_LABEL=USER_SUPPLIED\n", "UNKNOWN"),
        ("DATA_LABEL=REAL\x00\n", "UNKNOWN"),
        ("DATA_LABEL=UNKNOWN\n", "UNKNOWN"),
    ],
)
def test_extract_data_label(stdout: str, expected: str) -> None:
    assert extract_data_label(stdout) == expected


def test_real_sealed_serial_watch_receipt_verifies(
    tmp_path: Path, sealed_receipt_bytes: bytes
) -> None:
    from quant_fund.schemas.receipt_v2_leaf import verify_receipt_file

    path = _plant(tmp_path, sealed_receipt_bytes)
    assert verify_receipt_file(path)["valid"] is True
    item = verify_receipt(path, roots=[tmp_path])
    body = json.loads(sealed_receipt_bytes)
    assert item.verdict is Verdict.VERIFIED
    assert item.data_label == "SYNTHETIC"
    assert item.schema == "serial_watch.v1"
    assert item.seal == body["receipt_sha256"]
    assert item.sha256 == hashlib.sha256(sealed_receipt_bytes).hexdigest()
    assert item.ref == "serial_watch.json"
    assert item.item_id == "E0"
    assert item.detail == "sealed receipt verified"
    assert item.kind is EvidenceKind.RECEIPT


def test_tampered_sealed_receipt_is_invalid(tmp_path: Path, sealed_receipt_bytes: bytes) -> None:
    path = _plant(tmp_path, sealed_receipt_bytes)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["pooled_evalue"] = 12345.0
    path.write_text(json.dumps(body), encoding="utf-8")
    item = verify_receipt(path, roots=[tmp_path])
    assert item.verdict is Verdict.INVALID
    assert "receipt_sha256_mismatch" in item.detail
    assert item.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert item.verdict is not Verdict.VERIFIED


def test_missing_receipt(tmp_path: Path) -> None:
    calls: list[Path] = []

    def verifier(path: Path) -> Mapping[str, Any]:
        calls.append(path)
        return {"valid": True}

    item = verify_receipt(tmp_path / "missing.json", roots=[tmp_path], verifier=verifier)
    assert item.verdict is Verdict.MISSING
    assert item.detail == "receipt file not found"
    assert item.sha256 is None
    assert calls == []


def test_dotdot_escape_is_unverifiable_without_opening(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    calls: list[Path] = []

    def verifier(path: Path) -> Mapping[str, Any]:
        calls.append(path)
        return {"valid": True}

    item = verify_receipt(root / ".." / "nope.json", roots=[root], verifier=verifier)
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "outside allowed roots"
    assert item.sha256 is None
    assert calls == []


def test_symlink_escape_is_unverifiable_without_opening(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"data_label": "REAL"}', encoding="utf-8")
    link = root / "link.json"
    link.symlink_to(secret)
    calls: list[Path] = []

    def verifier(path: Path) -> Mapping[str, Any]:
        calls.append(path)
        return {"valid": True}

    item = verify_receipt(link, roots=[root], verifier=verifier)
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "outside allowed roots"
    assert calls == []
    assert item.sha256 is None
    assert "secret.json" not in item.ref


def test_symlink_inside_root_verifies_the_target(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = _write_json(root / "real.json", {"data_label": "REAL", "schema": "demo.v1"})
    link = root / "alias.json"
    link.symlink_to(target)
    seen: list[Path] = []

    def verifier(path: Path) -> Mapping[str, Any]:
        seen.append(path)
        return {"valid": True, "schema": "demo.v1"}

    item = verify_receipt(link, roots=[root], verifier=verifier)
    assert item.verdict is Verdict.VERIFIED
    assert seen == [target.resolve()]
    assert item.ref == "real.json"
    assert item.data_label == "REAL"


def test_oversize_receipt_is_not_opened(tmp_path: Path) -> None:
    path = tmp_path / "big.json"
    path.write_bytes(b"{" + b"a" * 80 + b"}")
    calls: list[Path] = []

    def verifier(path: Path) -> Mapping[str, Any]:
        calls.append(path)
        return {"valid": True}

    item = verify_receipt(path, roots=[tmp_path], max_bytes=10, verifier=verifier)
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "receipt exceeds size limit"
    assert calls == []
    assert item.sha256 is None


def test_receipt_at_size_limit_is_read(tmp_path: Path) -> None:
    path = tmp_path / "exact.json"
    payload = b"{}"
    path.write_bytes(payload)
    item = verify_receipt(path, roots=[tmp_path], max_bytes=len(payload), verifier=_ok_verifier())
    assert item.verdict is Verdict.VERIFIED
    assert item.sha256 == hashlib.sha256(payload).hexdigest()


def test_verifier_exception_is_unverifiable_and_redacted(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {"data_label": "REAL"})

    def verifier(path: Path) -> Mapping[str, Any]:
        del path
        raise RuntimeError("sk-supersecretkeyvalue123456")

    item = verify_receipt(path, roots=[tmp_path], verifier=verifier)
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail.startswith("verifier failed: RuntimeError")
    assert "sk-" not in item.detail
    assert "REDACTED" in item.detail
    assert item.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_verifier_exception_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.schemas.receipt_v2_leaf as leaf

    path = _write_json(tmp_path / "r.json", {})

    def boom(*_args: object, **_kwargs: object) -> Mapping[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(leaf, "verify_receipt_bytes", boom)
    item = verify_receipt(path, roots=[tmp_path])
    assert item.verdict is Verdict.UNVERIFIABLE
    assert "verifier failed: RuntimeError" in item.detail


def test_outside_path_does_not_call_default_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.schemas.receipt_v2_leaf as leaf

    calls: list[object] = []

    def boom(*args: object, **_kwargs: object) -> Mapping[str, Any]:
        calls.append(args)
        return {"valid": True}

    monkeypatch.setattr(leaf, "verify_receipt_bytes", boom)
    root = tmp_path / "root"
    root.mkdir()
    outside = _write_json(tmp_path / "out.json", {"data_label": "REAL"})
    item = verify_receipt(outside, roots=[root])
    assert item.verdict is Verdict.UNVERIFIABLE
    assert calls == []
    assert item.sha256 is None


def test_injectable_verifier_and_sha256(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "demo.json",
        {"data_label": "REAL", "receipt_sha256": HASH, "schema": "demo.v1"},
    )
    item = verify_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    assert item.verdict is Verdict.VERIFIED
    assert item.data_label == "REAL"
    assert item.schema == "demo.v1"
    assert item.seal == HASH
    assert item.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert item.sha256 != item.seal


def test_non_boolean_valid_is_unverifiable(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {})

    def verifier(path: Path) -> Mapping[str, Any]:
        del path
        return {"valid": "yes"}

    item = verify_receipt(path, roots=[tmp_path], verifier=verifier)
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "verifier returned no verdict"


def test_verifier_non_mapping_is_unverifiable(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {})
    item = verify_receipt(path, roots=[tmp_path], verifier=lambda _path: ["no"])  # type: ignore[arg-type,return-value]
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "verifier returned no verdict"


def test_invalid_errors_are_redacted(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {"data_label": "REAL"})

    def verifier(path: Path) -> Mapping[str, Any]:
        del path
        return {"valid": False, "errors": ["api_key=abcdef123456", "receipt_sha256_mismatch"]}

    item = verify_receipt(path, roots=[tmp_path], verifier=verifier)
    assert item.verdict is Verdict.INVALID
    assert "abcdef123456" not in item.detail
    assert "REDACTED" in item.detail
    assert "receipt_sha256_mismatch" in item.detail


def test_invalid_without_error_list(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {})
    item = verify_receipt(
        path,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": False, "errors": "nope"},
    )
    assert item.verdict is Verdict.INVALID
    assert item.detail == "receipt invalid"


def test_unknown_schema_falls_back_to_json(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {"schema": "demo.v1", "data_label": "REAL"})
    item = verify_receipt(
        path,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": True, "schema": "unknown"},
    )
    assert item.schema == "demo.v1"
    assert item.verdict is Verdict.VERIFIED


def test_verifier_seal_overrides_json_and_casefolds(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {"receipt_sha256": HASH})
    item = verify_receipt(
        path,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": True, "digest": OTHER_HASH.upper()},
    )
    assert item.seal == OTHER_HASH
    assert item.verdict is Verdict.VERIFIED


def test_nested_synthetic_label_is_conservative(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "r.json",
        {"data_label": "REAL", "payload": {"data_label": "SYNTHETIC", "kind": "inner.v1"}},
    )
    item = verify_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    assert item.data_label == "SYNTHETIC"


def test_top_level_label_used_when_nested_absent(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "r.json", {"data_label": "USER_SUPPLIED"})
    item = verify_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    assert item.data_label == "USER_SUPPLIED"


def test_relative_receipt_path_is_joined_to_root(tmp_path: Path) -> None:
    _write_json(tmp_path / "receipts" / "a.json", {"data_label": "USER_SUPPLIED"})
    item = verify_receipt("receipts/a.json", roots=[tmp_path], verifier=_ok_verifier())
    assert item.verdict is Verdict.VERIFIED
    assert item.ref == "receipts/a.json"


def test_unreadable_bytes_are_unverifiable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_json(tmp_path / "r.json", {})
    calls: list[Path] = []

    def boom(_path: Path, _max_bytes: int) -> bytes:
        raise OSError("denied")

    def verifier(receipt: Path) -> Mapping[str, Any]:
        calls.append(receipt)
        return {"valid": True}

    monkeypatch.setattr(evidence_module, "_read_limited", boom)
    item = verify_receipt(path, roots=[tmp_path], verifier=verifier)
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "receipt unreadable"
    assert calls == []
    assert item.sha256 is None


def test_directory_is_not_a_receipt(tmp_path: Path) -> None:
    item = verify_receipt(tmp_path, roots=[tmp_path])
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "not a regular file"


def test_empty_roots_are_outside(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {})
    item = verify_receipt(path, roots=[])
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "outside allowed roots"


def test_nul_path_is_rejected(tmp_path: Path) -> None:
    item = verify_receipt("a\x00b.json", roots=[tmp_path])
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.detail == "path rejected"


def test_non_posix_read_skips_nofollow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows has no O_NOFOLLOW; the read path must still verify a regular file."""
    monkeypatch.setattr(evidence_module, "_use_nofollow", lambda: False)
    path = _write_json(tmp_path / "r.json", {"data_label": "REAL"})
    item = verify_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    assert item.verdict is Verdict.VERIFIED
    assert item.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_ledger_assigns_ids_across_kinds(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "SYNTHETIC"})
    ledger = EvidenceLedger()
    web = ledger.add_web("https://example.com/a")
    receipt = ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier(), step="r-1.001")
    flash = ledger.add_flash("flash-1", verified_flag=False)
    assert [item.item_id for item in ledger.items()] == ["E1", "E2", "E3"]
    assert web.kind is EvidenceKind.WEB
    assert receipt.item_id == "E2"
    assert receipt.step == "r-1.001"
    assert flash.verdict is Verdict.UNVERIFIABLE
    other = EvidenceLedger()
    assert other.add_web("https://example.com/b").item_id == "E1"


def test_flash_flag_never_verifies() -> None:
    ledger = EvidenceLedger()
    flagged = ledger.add_flash("entry-1", verified_flag=True, detail="api_key=abcdef123456")
    plain = ledger.add_flash("entry-2", verified_flag=False)
    assert flagged.verdict is Verdict.UNVERIFIABLE
    assert plain.verdict is Verdict.UNVERIFIABLE
    assert "verified_flag" in flagged.detail
    assert "abcdef123456" not in flagged.detail
    assert "REDACTED" in flagged.detail
    text = ledger.render_block(width=200)
    assert "  VERIFIED  " not in text
    assert ledger.has_verified() is False
    assert ledger.verified_receipts() == []


def test_web_url_is_redacted_and_never_verified() -> None:
    ledger = EvidenceLedger()
    item = ledger.add_web("https://example.com/a?api_key=abcdef123456")
    assert item.verdict is Verdict.UNVERIFIABLE
    assert "abcdef123456" not in item.ref
    assert "REDACTED" in item.ref
    assert "abcdef123456" not in repr(item)
    assert item.detail == "HEURISTIC WEB VERIFICATION — not a sealed receipt"


def test_direct_non_receipt_verified_is_downgraded() -> None:
    item = EvidenceItem(
        item_id="E1",
        kind=EvidenceKind.MODEL,
        ref="turn",
        sha256=None,
        seal=None,
        verdict=Verdict.VERIFIED,
        data_label=" ",
        schema=None,
        detail="",
        step=None,
    )
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.data_label == "UNKNOWN"
    assert "non-receipt evidence cannot be VERIFIED" in item.detail


def test_ledger_labels_and_verified_receipts(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "SYNTHETIC"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    ledger.add_web("https://example.com/a")
    assert ledger.labels() == {"SYNTHETIC", "UNKNOWN"}
    assert len(ledger.verified_receipts()) == 1
    assert ledger.has_verified() is True


def test_ledger_roundtrip_and_next_id(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "REAL", "receipt_sha256": HASH})
    ledger = EvidenceLedger()
    ledger.add_receipt(
        path,
        roots=[tmp_path],
        verifier=_ok_verifier(),
        step="r-20261012T000000Z-abcdef01.001",
    )
    ledger.add_web("https://example.com/a")
    restored = EvidenceLedger.from_dict(ledger.to_dict())
    assert restored.to_dict() == ledger.to_dict()
    assert restored.items() == ledger.items()
    assert restored.add_web("https://example.com/b").item_id == "E3"


def test_from_dict_continues_after_highest_id(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    payload = ledger.to_dict()
    payload["items"][0]["item_id"] = "E4"
    restored = EvidenceLedger.from_dict(payload)
    assert restored.add_web("https://example.com").item_id == "E5"


def test_from_dict_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    payload = ledger.to_dict()
    payload["items"].append(dict(payload["items"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        EvidenceLedger.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"items": {}},
        {"items": ["nope"]},
        {"items": [{"item_id": "nope"}]},
    ],
)
def test_from_dict_rejects_bad_payloads(payload: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError):
        EvidenceLedger.from_dict(payload)


def test_from_dict_downgrades_forged_verified_web() -> None:
    raw = {
        "item_id": "E1",
        "kind": "WEB",
        "ref": "https://example.com",
        "sha256": None,
        "seal": None,
        "verdict": "VERIFIED",
        "data_label": "UNKNOWN",
        "schema": None,
        "detail": "nope",
        "step": None,
        "extra": "ignored",
    }
    ledger = EvidenceLedger.from_dict({"items": [raw]})
    item = ledger.items()[0]
    assert item.verdict is Verdict.UNVERIFIABLE
    assert "non-receipt evidence cannot be VERIFIED" in item.detail
    assert ledger.has_verified() is False
    again = EvidenceLedger.from_dict(ledger.to_dict())
    assert again.items()[0].detail == item.detail
    assert again.items()[0].verdict is Verdict.UNVERIFIABLE


def test_from_dict_rejects_short_sha(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    payload = ledger.to_dict()
    payload["items"][0]["sha256"] = "abcd"
    with pytest.raises(ValueError, match="sha256"):
        EvidenceLedger.from_dict(payload)


def test_concurrent_add_receipt_ids_are_unique(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait(timeout=2)
        ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert [item.item_id for item in ledger.items()] == [f"E{index}" for index in range(1, 9)]
    assert all(item.verdict is Verdict.VERIFIED for item in ledger.items())


def test_render_block_golden(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "receipts" / "vol_bench.json",
        {"data_label": "SYNTHETIC", "schema": "vol_bench.v1"},
    )
    ledger = EvidenceLedger()
    ledger.add_receipt(
        path,
        roots=[tmp_path],
        step="r-20261012T000000Z-abcdef01.001",
        verifier=_ok_verifier("vol_bench.v1"),
    )
    ledger.add_web("https://example.com/note")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    short = f"{digest[:4]}…{digest[-4:]}"
    expected = "\n".join(
        [
            "Evidence (harness-verified — authored by dipcatcher, not by the model)",
            (
                "  E1  receipt  VERIFIED  SYNTHETIC  vol_bench.v1  "
                f"sha256 {short}  receipts/vol_bench.json  "
                "step r-20261012T000000Z-abcdef01.001"
            ),
            (
                "  E2  web  UNVERIFIED  "
                "(heuristic web verification — not a sealed receipt)  "
                "https://example.com/note"
            ),
            (
                "Evidence class summary: 1 verified receipt "
                "(SYNTHETIC correctness evidence only — not market evidence); "
                "1 heuristic web source."
            ),
        ]
    )
    assert ledger.render_block(width=200) == expected


def test_render_block_passes_honesty_gate(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "SYNTHETIC", "schema": "demo.v1"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    ledger.add_web("https://example.com/note")
    ledger.add_flash("flash-1", verified_flag=True)
    block = ledger.render_block()
    assert validate_fx1_output(block) == block
    assert "\x1b" not in block
    assert block == ledger.render_block()
    wide = ledger.render_block(width=200)
    assert "SYNTHETIC correctness evidence only — not market evidence" in wide
    assert validate_fx1_output(wide) == wide


def test_render_unknown_real_and_mixed_synthetic_counts(tmp_path: Path) -> None:
    unknown = _write_json(tmp_path / "unknown.json", {})
    real = _write_json(tmp_path / "real.json", {"data_label": "REAL"})
    synthetic = _write_json(tmp_path / "synthetic.json", {"data_label": "SYNTHETIC"})
    only_unknown = EvidenceLedger()
    only_unknown.add_receipt(unknown, roots=[tmp_path], verifier=_ok_verifier())
    unknown_text = only_unknown.render_block(width=200)
    assert "UNKNOWN" in unknown_text
    assert "not market evidence" not in unknown_text
    only_real = EvidenceLedger()
    only_real.add_receipt(real, roots=[tmp_path], verifier=_ok_verifier())
    assert "Evidence class summary: 1 verified receipt." in only_real.render_block(width=200)
    mixed = EvidenceLedger()
    mixed.add_receipt(real, roots=[tmp_path], verifier=_ok_verifier())
    mixed.add_receipt(synthetic, roots=[tmp_path], verifier=_ok_verifier())
    mixed_text = mixed.render_block(width=200)
    assert (
        "2 verified receipts (1 SYNTHETIC correctness evidence only — not market evidence)"
        in mixed_text
    )
    both = EvidenceLedger()
    both.add_receipt(synthetic, roots=[tmp_path], verifier=_ok_verifier())
    other = _write_json(tmp_path / "synthetic-2.json", {"data_label": "SYNTHETIC"})
    both.add_receipt(other, roots=[tmp_path], verifier=_ok_verifier())
    assert (
        "2 verified receipts (SYNTHETIC correctness evidence only — not market evidence)"
        in both.render_block(width=200)
    )
    assert validate_fx1_output(mixed_text) == mixed_text


def test_render_names_invalid_missing_and_unverified(tmp_path: Path) -> None:
    ledger = EvidenceLedger()
    ledger.add_receipt(tmp_path / "missing.json", roots=[tmp_path])
    bad = _write_json(tmp_path / "bad.json", {"data_label": "SYNTHETIC"})
    ledger.add_receipt(
        bad,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": False, "errors": ["receipt_sha256_mismatch"]},
    )
    ledger.add_receipt(tmp_path.parent / "outside.json", roots=[tmp_path])
    block = ledger.render_block(width=240)
    assert (
        "Evidence class summary: 0 verified receipts; 1 INVALID receipt; "
        "1 MISSING receipt; 1 UNVERIFIED receipt."
    ) in block
    assert "outside allowed roots" in block
    assert "receipt_sha256_mismatch" in block
    assert validate_fx1_output(block) == block


def test_render_empty_ledger() -> None:
    text = EvidenceLedger().render_block()
    assert text == (
        "Evidence (harness-verified — authored by dipcatcher, not by the model)\n"
        "Evidence class summary: no evidence items."
    )
    assert validate_fx1_output(text) == text


def test_render_wraps_long_lines() -> None:
    ledger = EvidenceLedger()
    ledger.add_web("https://example.com/" + ("segment" * 30))
    text = ledger.render_block(width=40)
    assert all(len(line) <= 40 for line in text.splitlines())
    packed = EvidenceLedger()
    packed.add_web("https://example.com/" + ("a" * 80))
    packed_text = packed.render_block(width=24)
    assert all(len(line) <= 24 for line in packed_text.splitlines())
    narrow = EvidenceLedger().render_block(width=10)
    assert all(len(line) <= 16 for line in narrow.splitlines())


def test_render_model_item_is_not_verified() -> None:
    ledger = EvidenceLedger.from_dict(
        {
            "items": [
                {
                    "item_id": "E1",
                    "kind": "MODEL",
                    "ref": "turn-1",
                    "sha256": None,
                    "seal": None,
                    "verdict": "UNVERIFIABLE",
                    "data_label": "UNKNOWN",
                    "schema": None,
                    "detail": "model note",
                    "step": None,
                }
            ]
        }
    )
    text = ledger.render_block(width=200)
    assert "  E1  model  UNVERIFIED  (model interpretation — not a sealed receipt)  turn-1" in text
    assert "1 model item." in text
    assert ledger.has_verified() is False


@pytest.mark.parametrize(
    ("answer", "codes"),
    [
        ("no hashes here", []),
        ("unverified prose", []),
        ("unproven claim", []),
        ("this is verified", ["verified_language_without_receipt"]),
        ("Proven by the run", ["verified_language_without_receipt"]),
        ("Confirmed by receipt today", ["verified_language_without_receipt"]),
        (HASH, ["unknown_hash"]),
        (HASH.upper(), ["unknown_hash"]),
        (HASH[:63], []),
        (HASH + "aa", []),
        ("see file.json.bak please", []),
        ("see receipts/missing.json", ["unknown_receipt_path"]),
        ("vol_bench.v1 is a schema", []),
    ],
)
def test_check_citations_without_verified_receipt(answer: str, codes: list[str]) -> None:
    problems = check_citations(answer, EvidenceLedger())
    assert [problem.code for problem in problems] == codes


def test_citations_accept_verified_seal_case_insensitively(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "ok.json", {"data_label": "REAL", "receipt_sha256": HASH})
    ledger = EvidenceLedger()
    item = ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    assert item.sha256 is not None
    assert check_citations(HASH.upper(), ledger) == []
    assert check_citations(item.sha256.upper(), ledger) == []
    assert check_citations("the result is verified", ledger) == []


def test_citations_flag_unverified_hash(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "bad.json", {"data_label": "REAL", "receipt_sha256": HASH})
    ledger = EvidenceLedger()
    item = ledger.add_receipt(
        path,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": False, "errors": ["receipt_sha256_mismatch"]},
    )
    problems = check_citations(HASH, ledger)
    assert [problem.code for problem in problems] == ["unverified_hash"]
    assert item.item_id in problems[0].detail
    assert "INVALID" in problems[0].detail


def test_verified_hash_wins_over_duplicate_unverified_seal(tmp_path: Path) -> None:
    good = _write_json(tmp_path / "good.json", {"data_label": "REAL", "receipt_sha256": HASH})
    bad = _write_json(tmp_path / "bad.json", {"data_label": "REAL", "receipt_sha256": HASH})
    ledger = EvidenceLedger()
    ledger.add_receipt(good, roots=[tmp_path], verifier=_ok_verifier())
    ledger.add_receipt(
        bad,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": False, "errors": ["receipt_sha256_mismatch"]},
    )
    assert check_citations(HASH, ledger) == []


def test_citations_match_receipt_paths(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "receipts" / "vol_bench.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    assert check_citations("see vol_bench.json", ledger) == []
    assert check_citations(f"see {path}", ledger) == []
    assert check_citations("see receipts\\vol_bench.json", ledger) == []
    problems = check_citations("see other/vol_bench.json", ledger)
    assert [problem.code for problem in problems] == ["unknown_receipt_path"]


def test_citation_problem_order(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "bad.json", {"data_label": "SYNTHETIC"})
    ledger = EvidenceLedger()
    ledger.add_receipt(
        path,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": False, "errors": ["receipt_sha256_mismatch"]},
    )
    answer = f"verified {OTHER_HASH} receipts/missing.json and synthetic only"
    problems = check_citations(answer, ledger)
    assert [problem.code for problem in problems] == [
        "unknown_hash",
        "unknown_receipt_path",
        "verified_language_without_receipt",
        "label_missing",
    ]


def test_lowercase_synthetic_does_not_satisfy_label(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "SYNTHETIC"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    problems = check_citations("synthetic note", ledger)
    assert [problem.code for problem in problems] == ["label_missing"]
    assert check_citations("SYNTHETIC note", ledger) == []


def test_duplicate_hash_is_reported_once() -> None:
    problems = check_citations(f"{HASH} and again {HASH}", EvidenceLedger())
    assert [problem.code for problem in problems] == ["unknown_hash"]


def test_compose_verified_real_receipt(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "ok.json", {"data_label": "REAL", "receipt_sha256": HASH})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    answer = f"Interval pinball is discussed against {HASH}."
    out = compose_answer(answer, ledger)
    assert out.status == "verified"
    assert out.problems == ()
    assert out.text.startswith("Model interpretation (cites harness-verified evidence below):")
    assert answer in out.text
    assert out.text.index(answer) < out.text.index("Evidence (harness-verified")
    assert validate_fx1_output(out.text) == out.text


def test_compose_real_sealed_receipt(tmp_path: Path, sealed_receipt_bytes: bytes) -> None:
    path = _plant(tmp_path, sealed_receipt_bytes)
    ledger = EvidenceLedger()
    item = ledger.add_receipt(path, roots=[tmp_path])
    assert item.verdict is Verdict.VERIFIED
    assert item.seal is not None
    answer = f"SYNTHETIC serial check sealed by {item.seal} in {path.name}."
    out = compose_answer(answer, ledger)
    assert out.status == "verified"
    assert out.problems == ()
    assert validate_fx1_output(out.text) == out.text
    assert "not market evidence" in out.text.replace("\n    ", " ")


def test_compose_partial_when_web_accompanies_verified_receipt(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "ok.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    ledger.add_web("https://example.com/note")
    out = compose_answer("pinball only", ledger)
    assert out.status == "partial"
    assert out.problems == ()
    assert out.text.startswith("Model interpretation (cites harness-verified evidence below):")
    assert "heuristic web verification" in out.text
    assert "pinball only" in out.text


def test_compose_partial_lists_citation_warnings(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "ok.json", {"data_label": "REAL", "receipt_sha256": HASH})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    answer = f"pinball only {OTHER_HASH}"
    out = compose_answer(answer, ledger)
    assert out.status == "partial"
    assert out.problems[0].code == "unknown_hash"
    assert out.text.startswith("Model interpretation (not independently verified):")
    assert "Citation warnings:" in out.text
    assert out.text.index("Citation warnings:") < out.text.index("Evidence (harness-verified")
    assert answer in out.text


def test_compose_unverified_when_receipt_is_invalid(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "bad.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    ledger.add_receipt(
        path,
        roots=[tmp_path],
        verifier=lambda _path: {"valid": False, "errors": ["receipt_sha256_mismatch"]},
    )
    out = compose_answer("no hash here", ledger)
    assert out.status == "unverified"
    assert "no hash here" in out.text
    assert "INVALID" in out.text
    assert out.text.startswith("Model interpretation (not independently verified):")


def test_compose_no_evidence() -> None:
    out = compose_answer("hello", EvidenceLedger())
    assert out.status == "no-evidence"
    assert "hello" in out.text
    assert "no evidence items" in out.text
    assert out.text.startswith("Model interpretation (not independently verified):")
    assert out.problems == ()


def test_compose_no_evidence_still_reports_verified_language() -> None:
    out = compose_answer("this is verified", EvidenceLedger())
    assert out.status == "no-evidence"
    assert any(problem.code == "verified_language_without_receipt" for problem in out.problems)
    assert "Citation warnings:" in out.text
    assert "this is verified" in out.text


def test_compose_synthetic_label_blocks_verified_status(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "a.json", {"data_label": "SYNTHETIC", "receipt_sha256": HASH})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())
    missing = compose_answer(f"sealed {HASH}", ledger)
    assert missing.status == "partial"
    assert any(problem.code == "label_missing" for problem in missing.problems)
    present = compose_answer(f"SYNTHETIC sealed {HASH}", ledger)
    assert present.status == "verified"
    assert present.problems == ()


def test_compose_withheld_on_injected_gate(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "ok.json", {"data_label": "REAL"})
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path], verifier=_ok_verifier())

    def reject(text: str) -> str:
        del text
        raise Fx1HonestyError("api_key=abcdef123456")

    out = compose_answer("secret prose 3.5", ledger, validate=reject)
    assert out.status == "withheld"
    assert "secret prose" not in out.text
    assert "3.5" not in out.text
    assert "abcdef123456" not in out.text
    assert "REDACTED" in out.text
    assert "Evidence (harness-verified" in out.text
    assert validate_fx1_output(out.text) == out.text


def test_compose_withheld_on_honesty_gate() -> None:
    out = compose_answer("The sharpe ratio is 3.5", EvidenceLedger())
    assert out.status == "withheld"
    assert "3.5" not in out.text
    assert "The sharpe ratio" not in out.text
    assert validate_fx1_output(out.text) == out.text
    live = compose_answer("guaranteed returns on this book", EvidenceLedger())
    assert live.status == "withheld"
    assert "guaranteed returns" not in live.text
    synthetic = compose_answer("synthetic results show a score of 0.2", EvidenceLedger())
    assert synthetic.status == "withheld"
    assert "0.2" not in synthetic.text


def test_compose_checks_citations_on_validated_text() -> None:
    def rewrite(text: str) -> str:
        del text
        return "this is verified"

    out = compose_answer("plain", EvidenceLedger(), validate=rewrite)
    assert out.status == "no-evidence"
    assert any(problem.code == "verified_language_without_receipt" for problem in out.problems)
    assert "this is verified" in out.text
    assert "plain" not in out.text


def test_compose_withheld_citations_use_original_text() -> None:
    def reject(text: str) -> str:
        del text
        raise Fx1HonestyError("gate")

    out = compose_answer("this is verified", EvidenceLedger(), validate=reject)
    assert out.status == "withheld"
    assert "this is verified" not in out.text
    assert any(problem.code == "verified_language_without_receipt" for problem in out.problems)


def test_exit_zero_wording_does_not_verify() -> None:
    ledger = EvidenceLedger()
    ledger.add_web("https://example.com/note")
    out = compose_answer("command exit 0, result verified", ledger)
    assert out.status != "verified"
    assert out.status == "unverified"
    assert any(problem.code == "verified_language_without_receipt" for problem in out.problems)


def test_evidence_module_does_not_import_research() -> None:
    source = Path(evidence_module.__file__ or "").read_text(encoding="utf-8")
    assert "quant_fund.research" not in source
    assert "verify_receipt_bytes" in source


def test_citation_problem_is_frozen() -> None:
    problem = CitationProblem(code="unknown_hash", detail="x")
    with pytest.raises(AttributeError):
        problem.code = "other"  # type: ignore[misc]
