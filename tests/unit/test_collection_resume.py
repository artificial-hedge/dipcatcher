"""Resume must not treat an interrupted collection write as a finished file.

SYNTHETIC frames only: these tests prove the resume decision, not market data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "collect_perp_universe.py"

_spec = importlib.util.spec_from_file_location("collect_perp_universe", SCRIPT)
assert _spec is not None and _spec.loader is not None
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)

pytestmark = pytest.mark.synthetic


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1), datetime(2024, 1, 1, 1)],
            "close": [1.0, 2.0],
        }
    )


def _seal(path: Path, frame: pl.DataFrame | None = None) -> str:
    (frame if frame is not None else _frame()).write_parquet(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"schema_version": 2, "sha256": digest, "rows": 2}
    path.with_suffix(".json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return digest


def test_sealed_parquet_is_reused_and_audited(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_1h.perp.parquet"
    digest = _seal(path)
    manifest: dict = {"files": {}, "skipped_existing": []}
    assert collect.existing_output_action(path, resume=True) == "reuse"
    collect._accept_reusable(manifest, path.name, path, "1h")
    assert manifest["skipped_existing"] == [path.name]
    assert manifest["files"][path.name]["sha256"] == digest
    assert manifest["files"][path.name]["rows"] == 2


def test_missing_and_zero_length_files_are_collected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    empty = tmp_path / "empty.parquet"
    empty.write_bytes(b"")
    assert collect.existing_output_action(missing, resume=True) == "collect"
    assert collect.existing_output_action(empty, resume=True) == "collect"


def test_orphan_temp_file_is_not_a_finished_output(tmp_path: Path) -> None:
    (tmp_path / "btcusdt_1h.perp.parquet.tmp").write_bytes(b"partial-bytes")
    target = tmp_path / "btcusdt_1h.perp.parquet"
    assert collect.existing_output_action(target, resume=True) == "collect"


def test_nonempty_file_without_receipt_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_1h.perp.parquet"
    payload = b"partial-output-not-empty"
    path.write_bytes(payload)
    with pytest.raises(collect.PartialCollectionError, match="readable parquet"):
        collect.existing_output_action(path, resume=True)
    assert path.read_bytes() == payload


def test_readable_parquet_without_receipt_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_1h.perp.parquet"
    _frame().write_parquet(path)
    before = path.read_bytes()
    with pytest.raises(collect.PartialCollectionError, match="no receipt"):
        collect.existing_output_action(path, resume=True)
    assert path.read_bytes() == before


def test_receipt_hash_mismatch_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_1h.perp.parquet"
    _frame().write_parquet(path)
    path.with_suffix(".json").write_text(
        json.dumps({"sha256": "0" * 64}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(collect.PartialCollectionError, match="does not match"):
        collect.existing_output_action(path, resume=True)


def test_torn_receipt_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_1h.perp.parquet"
    _frame().write_parquet(path)
    path.with_suffix(".json").write_text("{", encoding="utf-8")
    with pytest.raises(collect.PartialCollectionError, match="unreadable"):
        collect.existing_output_action(path, resume=True)


def test_hash_match_on_unreadable_bytes_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_1h.perp.parquet"
    blob = b"not-a-parquet-frame"
    path.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    path.with_suffix(".json").write_text(json.dumps({"sha256": digest}) + "\n", encoding="utf-8")
    with pytest.raises(collect.PartialCollectionError, match="readable parquet"):
        collect.existing_output_action(path, resume=True)


def test_explicit_recollect_does_not_trust_a_partial(tmp_path: Path) -> None:
    path = tmp_path / "btcusdt_1h.perp.parquet"
    path.write_bytes(b"partial")
    assert collect.existing_output_action(path, resume=False) == "collect"


def test_shard_refuses_torn_universe_and_keeps_a_readable_one(tmp_path: Path) -> None:
    torn = tmp_path / "torn.parquet"
    torn.write_bytes(b"torn-parquet")
    shard = pl.DataFrame({"security_id": ["BTCUSDT"]})
    with pytest.raises(collect.PartialCollectionError, match="readable parquet"):
        collect._publish_universe(shard, torn, keep_existing=True)
    assert torn.read_bytes() == b"torn-parquet"

    kept = tmp_path / "perp_universe.parquet"
    original = pl.DataFrame({"security_id": ["BTCUSDT", "ETHUSDT"]})
    original.write_parquet(kept)
    before = kept.read_bytes()
    collect._publish_universe(shard, kept, keep_existing=True)
    assert kept.read_bytes() == before
    assert pl.read_parquet(kept)["security_id"].to_list() == ["BTCUSDT", "ETHUSDT"]


def test_universe_replace_is_readable_parquet(tmp_path: Path) -> None:
    path = tmp_path / "perp_universe.parquet"
    path.write_bytes(b"torn")
    frame = pl.DataFrame({"security_id": ["BTCUSDT"]})
    collect._publish_universe(frame, path, keep_existing=False)
    assert pl.read_parquet(path)["security_id"].to_list() == ["BTCUSDT"]


def test_manifest_publish_replaces_torn_text(tmp_path: Path) -> None:
    path = tmp_path / "collection_manifest.json"
    path.write_text("{torn", encoding="utf-8")
    payload = {
        "failures": {},
        "files": {"a.parquet": {"sha256": "abc"}},
        "schema": "perp_collection.v1",
    }
    collect._publish_manifest(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert [item for item in path.parent.iterdir() if item != path] == []


def test_resume_rejects_symlink_output(tmp_path: Path) -> None:
    target = tmp_path / "real.parquet"
    target.write_bytes(b"not-empty")
    link = tmp_path / "btcusdt_1h.perp.parquet"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not available")
    with pytest.raises(collect.PartialCollectionError, match="regular file"):
        collect.existing_output_action(link, resume=True)
    dangling = tmp_path / "dangling.parquet"
    try:
        dangling.symlink_to(tmp_path / "missing.parquet")
    except OSError:
        pytest.skip("symlinks are not available")
    with pytest.raises(collect.PartialCollectionError, match="regular file"):
        collect.existing_output_action(dangling, resume=True)
