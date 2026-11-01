"""Coverage for ``utils/receipt.py`` — the shared receipt-seal primitive.

The seal is wrapped by lane packages, but the quarantine rules for the corpus
listing (``is_quarantined`` / ``verified_corpus_files``) are only reached
through research scanners, so they are pinned directly here.
"""

from __future__ import annotations

from pathlib import Path

from quant_fund.utils.receipt import (
    is_quarantined,
    seal_errors,
    seal_receipt,
    verified_corpus_files,
)


def _receipt() -> dict[str, object]:
    return {"claim": "SYNTHETIC", "score": 0.5, "meta": {"lane": "unit"}}


def test_seal_receipt_binds_a_digest_and_is_deterministic() -> None:
    first = seal_receipt(_receipt())
    second = seal_receipt(_receipt())
    assert isinstance(first["receipt_sha256"], str)
    assert len(first["receipt_sha256"]) == 64
    assert first == second
    # The digest covers every unsigned field, not just a subset of them.
    assert seal_receipt({**_receipt(), "score": 0.6})["receipt_sha256"] != first["receipt_sha256"]


def test_seal_receipt_replaces_a_stale_digest() -> None:
    sealed = seal_receipt({**_receipt(), "receipt_sha256": "0" * 64})
    assert sealed["receipt_sha256"] != "0" * 64
    assert seal_errors(sealed) == []


def test_seal_receipt_exclusions_ignore_post_seal_keys() -> None:
    base = seal_receipt(_receipt())
    with_paths = seal_receipt({**_receipt(), "path": "/tmp/x.json"}, exclude=("path",))
    assert with_paths["receipt_sha256"] == base["receipt_sha256"]
    assert seal_errors(with_paths, exclude=("path",)) == []
    assert seal_errors(with_paths) == ["receipt_sha256"]


def test_seal_errors_detects_tampering_and_missing_digest() -> None:
    sealed = seal_receipt(_receipt())
    assert seal_errors(sealed) == []
    assert seal_errors({**sealed, "score": 0.6}) == ["receipt_sha256"]
    assert seal_errors(_receipt()) == ["receipt_sha256"]


def test_is_quarantined_covers_nested_and_direct_children(tmp_path: Path) -> None:
    quarantined = tmp_path / "legacy-unsealed" / "old.json"
    direct = tmp_path / "claim.json"
    nested_elsewhere = tmp_path / "lane" / "legacy-unsealed" / "x.json"
    assert is_quarantined(quarantined, tmp_path) is True
    assert is_quarantined(direct, tmp_path) is False
    # Only a top-level quarantine directory is special, not a nested namesake.
    assert is_quarantined(nested_elsewhere, tmp_path) is False


def test_is_quarantined_fails_closed_for_paths_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "elsewhere" / "claim.json"
    assert is_quarantined(outside, tmp_path) is True


def test_verified_corpus_files_skips_seals_and_quarantine(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "a.json.seal.json").write_text("{}")
    (tmp_path / "lane").mkdir()
    (tmp_path / "lane" / "c.json").write_text("{}")
    quarantine = tmp_path / "legacy-unsealed"
    quarantine.mkdir()
    (quarantine / "old.json").write_text("{}")

    found = verified_corpus_files(tmp_path)

    assert found == sorted([tmp_path / "a.json", tmp_path / "b.json", tmp_path / "lane" / "c.json"])


def test_verified_corpus_files_honours_pattern(tmp_path: Path) -> None:
    (tmp_path / "one.json").write_text("{}")
    (tmp_path / "two.txt").write_text("")
    (tmp_path / "legacy-unsealed").mkdir()
    (tmp_path / "legacy-unsealed" / "three.txt").write_text("")

    assert verified_corpus_files(tmp_path, pattern="*.txt") == [tmp_path / "two.txt"]


def test_verified_corpus_files_empty_root(tmp_path: Path) -> None:
    assert verified_corpus_files(tmp_path) == []
