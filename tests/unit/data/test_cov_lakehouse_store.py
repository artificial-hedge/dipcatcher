"""Fail-closed branches of the content-addressed lake store.

Every write path is immutable and fail-closed: an object or snapshot that
already exists with different bytes, a symlink where a regular file belongs,
or a manifest field that cannot be a path token must raise ``DataContractError``
instead of overwriting or silently normalizing. All frames here are SYNTHETIC.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant_fund.data.lakehouse import store as store_mod
from quant_fund.data.lakehouse.store import (
    SNAPSHOT_SCHEMA,
    FileEntry,
    commit_snapshot,
    file_entry_for_object,
    load_snapshot,
    object_relpath,
    put_file,
    snapshot_id_for,
    write_partitioned_bars,
)
from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.reproducibility import content_address

DAY = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "symbol": ["AAPL", "AAPL"],
            "event_time": [DAY, DAY],
            "source": ["stooq", "stooq"],
            "open": [10.0, 10.5],
            "high": [11.0, 11.0],
            "low": [9.5, 10.0],
            "close": [10.8, 10.9],
            "volume": [100.0, 120.0],
        }
    )


def _entry_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "logical_path": "partitions/source=stooq/symbol=AAPL/date=2024-01-02/part.parquet",
        "object_path": "objects/sha256/ab/" + "ab" * 32,
        "content_sha256": "ab" * 32,
        "stored_sha256": "ab" * 32,
        "stored_size": 10,
        "declared_size": 10,
        "kind": "blob",
        "role": "bars",
        "source": "stooq",
        "symbol": "AAPL",
        "date": "2024-01-02",
        "byte_identical_to_source": True,
    }
    base.update(overrides)
    return base


# --- FileEntry.from_dict validation ---------------------------------------


def test_file_entry_requires_every_field() -> None:
    with pytest.raises(DataContractError, match="malformed"):
        FileEntry.from_dict({"logical_path": "a/b"})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"byte_identical_to_source": "yes"}, "must be a bool"),
        ({"byte_identical_to_source": 1}, "must be a bool"),
        ({"stored_size": -1}, "stored_size must be a non-negative integer"),
        ({"stored_size": True}, "stored_size must be a non-negative integer"),
        ({"declared_size": 1.5}, "declared_size must be a non-negative integer"),
        ({"kind": "tar"}, "blob or lfs_pointer"),
        ({"content_sha256": "AB" * 32}, "content_sha256 must be a lowercase sha256"),
        ({"content_sha256": "ab" * 31}, "content_sha256 must be a lowercase sha256"),
        ({"stored_sha256": 7}, "stored_sha256 must be a lowercase sha256"),
        ({"role": "bad/role"}, "unsafe role"),
        ({"source": "bad source"}, "unsafe source"),
        ({"symbol": "a b"}, "unsafe symbol"),
        ({"date": "2024-1-2"}, "must be YYYY-MM-DD"),
        ({"date": "2024-13-01"}, "must be YYYY-MM-DD"),
        ({"logical_path": "/abs/part.parquet"}, "must be a relative path"),
        ({"logical_path": ""}, "must be a relative path"),
        ({"object_path": "objects/../escape"}, "must stay inside the lake"),
        ({"object_path": "objects\\x"}, "must stay inside the lake"),
    ],
)
def test_file_entry_rejects_malformed_fields(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(DataContractError, match=match):
        FileEntry.from_dict(_entry_dict(**overrides))


def test_file_entry_allows_absent_optional_fields_and_round_trips() -> None:
    raw = _entry_dict(source=None, symbol=None, date=None)
    entry = FileEntry.from_dict(raw)
    assert entry.source is None
    assert entry.date is None
    assert entry.to_dict() == raw


# --- snapshot manifest validation -----------------------------------------


@pytest.mark.parametrize("dataset", ["", "   ", "bars\nillegal", "bars\x00id"])
def test_commit_snapshot_rejects_unsafe_dataset(tmp_path: Path, dataset: str) -> None:
    with pytest.raises(DataContractError, match="dataset name must be a non-empty single line"):
        commit_snapshot(tmp_path, dataset, [])


def test_commit_snapshot_rejects_duplicate_logical_paths(tmp_path: Path) -> None:
    entry = FileEntry.from_dict(_entry_dict())
    with pytest.raises(DataContractError, match="logical paths must be unique"):
        commit_snapshot(tmp_path, "bars", [entry, entry])


def test_commit_snapshot_refuses_a_symlinked_snapshot(tmp_path: Path) -> None:
    snapshot_id = snapshot_id_for("bars", ())
    directory = tmp_path / "snapshots"
    directory.mkdir(parents=True)
    (directory / f"{snapshot_id}.json").symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(DataContractError, match="is a symlink"):
        commit_snapshot(tmp_path, "bars", [])


def test_commit_snapshot_refuses_existing_bytes_that_differ(tmp_path: Path) -> None:
    snapshot_id = snapshot_id_for("bars", ())
    directory = tmp_path / "snapshots"
    directory.mkdir(parents=True)
    (directory / f"{snapshot_id}.json").write_bytes(b'{"schema":"tampered"}\n')
    with pytest.raises(DataContractError, match="already exists with different bytes"):
        commit_snapshot(tmp_path, "bars", [])


def test_commit_snapshot_refuses_a_racing_writer_with_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def racing_link(source: object, destination: object) -> None:
        Path(str(destination)).write_bytes(b"other writer won\n")
        raise FileExistsError(str(destination))

    monkeypatch.setattr(store_mod.os, "link", racing_link)
    with pytest.raises(DataContractError, match="immutable lake record"):
        commit_snapshot(tmp_path, "bars", [])


def test_load_snapshot_rejects_malformed_json(tmp_path: Path) -> None:
    digest = "cd" * 32
    directory = tmp_path / "snapshots"
    directory.mkdir(parents=True)
    (directory / f"{digest}.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DataContractError, match="is malformed"):
        load_snapshot(tmp_path, digest)


def test_load_snapshot_rejects_unsupported_schema(tmp_path: Path) -> None:
    digest = "cd" * 32
    directory = tmp_path / "snapshots"
    directory.mkdir(parents=True)
    body = {"schema": "dipcatcher.lake.snapshot.v0", "dataset": "bars", "files": []}
    (directory / f"{digest}.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(DataContractError, match="unsupported schema"):
        load_snapshot(tmp_path, digest)


@pytest.mark.parametrize(
    "body",
    [
        {"schema": SNAPSHOT_SCHEMA, "dataset": 3, "files": []},
        {"schema": SNAPSHOT_SCHEMA, "dataset": "bars", "files": {}},
        {"schema": SNAPSHOT_SCHEMA, "dataset": None, "files": []},
    ],
)
def test_load_snapshot_rejects_wrong_types(tmp_path: Path, body: object) -> None:
    digest = "cd" * 32
    directory = tmp_path / "snapshots"
    directory.mkdir(parents=True)
    (directory / f"{digest}.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(DataContractError, match="is malformed"):
        load_snapshot(tmp_path, digest)


def test_load_snapshot_rejects_a_null_manifest(tmp_path: Path) -> None:
    digest = "cd" * 32
    directory = tmp_path / "snapshots"
    directory.mkdir(parents=True)
    (directory / f"{digest}.json").write_text("[]", encoding="utf-8")
    with pytest.raises(DataContractError, match="unsupported schema"):
        load_snapshot(tmp_path, digest)


def test_load_snapshot_rejects_a_non_object_file_entry(tmp_path: Path) -> None:
    digest = "cd" * 32
    directory = tmp_path / "snapshots"
    directory.mkdir(parents=True)
    body = {"schema": SNAPSHOT_SCHEMA, "dataset": "bars", "files": ["nope"]}
    (directory / f"{digest}.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(DataContractError, match="non-object file entry"):
        load_snapshot(tmp_path, digest)


def test_object_relpath_rejects_non_hex_digest() -> None:
    with pytest.raises(DataContractError, match="stored_sha256 must be a lowercase sha256"):
        object_relpath("zz" * 32)


# --- object store write paths ---------------------------------------------


def test_put_file_refuses_a_symlinked_object(tmp_path: Path) -> None:
    source = tmp_path / "part.parquet"
    source.write_bytes(b"synthetic-bytes")
    destination = tmp_path / object_relpath(content_address(source).stored_sha256)
    destination.parent.mkdir(parents=True)
    destination.symlink_to(source)
    with pytest.raises(DataContractError, match="is a symlink"):
        put_file(tmp_path, source)


def test_put_file_refuses_existing_bytes_that_differ(tmp_path: Path) -> None:
    source = tmp_path / "part.parquet"
    source.write_bytes(b"synthetic-bytes")
    digest = content_address(source).stored_sha256
    destination = tmp_path / object_relpath(digest)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"a different object")
    with pytest.raises(DataContractError, match="exists with different bytes"):
        put_file(tmp_path, source)


def test_put_file_refuses_bytes_truncated_in_transit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "part.parquet"
    source.write_bytes(b"synthetic-bytes")
    monkeypatch.setattr(
        store_mod, "_copy_exact", lambda _src, dst: Path(dst).write_bytes(b"partial")
    )
    with pytest.raises(DataContractError, match="copied object bytes do not match"):
        put_file(tmp_path, source)


def test_put_file_refuses_an_object_path_that_is_not_a_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "part.parquet"
    source.write_bytes(b"synthetic-bytes")
    destination = tmp_path / object_relpath(content_address(source).stored_sha256)
    destination.mkdir(parents=True)
    with pytest.raises(DataContractError, match="is not a regular file"):
        put_file(tmp_path, source)


def test_put_file_refuses_a_racing_writer_with_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "part.parquet"
    source.write_bytes(b"synthetic-bytes")

    def racing_link(src: object, dst: object) -> None:
        Path(str(dst)).write_bytes(b"other writer won")
        raise FileExistsError(str(dst))

    monkeypatch.setattr(store_mod.os, "link", racing_link)
    with pytest.raises(DataContractError, match="exists with different bytes"):
        put_file(tmp_path, source)


def test_put_file_is_idempotent_for_identical_bytes(tmp_path: Path) -> None:
    source = tmp_path / "part.parquet"
    source.write_bytes(b"synthetic-bytes")
    first, address = put_file(tmp_path, source)
    second, again = put_file(tmp_path, source)
    assert first == second
    assert again.stored_sha256 == address.stored_sha256
    assert first.read_bytes() == b"synthetic-bytes"


# --- partitioned bar writes -----------------------------------------------


def test_write_partitioned_bars_rejects_missing_columns(tmp_path: Path) -> None:
    frame = _bars().drop("close")
    with pytest.raises(DataContractError, match=r"missing columns: \['close'\]"):
        write_partitioned_bars(tmp_path, frame, dataset="bars")


def test_write_partitioned_bars_requires_symbol_or_security_id(tmp_path: Path) -> None:
    frame = _bars().drop(["symbol", "security_id"])
    with pytest.raises(DataContractError, match="need symbol or security_id"):
        write_partitioned_bars(tmp_path, frame, dataset="bars")


def test_write_partitioned_bars_falls_back_to_security_id(tmp_path: Path) -> None:
    snapshot = write_partitioned_bars(tmp_path, _bars().drop("symbol"), dataset="bars")
    assert len(snapshot.files) == 1
    assert snapshot.files[0].symbol == "A"


def test_write_partitioned_bars_rejects_an_empty_frame(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="non-empty frame"):
        write_partitioned_bars(tmp_path, _bars().head(0), dataset="bars")


def test_write_partitioned_bars_requires_utc_event_time(tmp_path: Path) -> None:
    naive = _bars().with_columns(pl.col("event_time").dt.replace_time_zone(None))
    with pytest.raises(DataContractError, match="event_time must be a UTC datetime"):
        write_partitioned_bars(tmp_path, naive, dataset="bars")


def test_write_partitioned_bars_rejects_unsafe_source_tokens(tmp_path: Path) -> None:
    frame = _bars().with_columns(pl.lit("bad/source").alias("source"))
    with pytest.raises(DataContractError, match="unsafe source"):
        write_partitioned_bars(tmp_path, frame, dataset="bars")


def test_write_partitioned_bars_rejects_unsafe_symbol_tokens(tmp_path: Path) -> None:
    frame = _bars().with_columns(pl.lit("A B").alias("symbol"))
    with pytest.raises(DataContractError, match="unsafe symbol"):
        write_partitioned_bars(tmp_path, frame, dataset="bars")


def test_write_partitioned_bars_rejects_a_malformed_partition_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pl.DataFrame,
        "partition_by",
        lambda self, *args, **kwargs: {"stooq": self},  # noqa: ARG005
    )
    with pytest.raises(DataContractError, match="unexpected partition key"):
        write_partitioned_bars(tmp_path, _bars(), dataset="bars")


# --- helper edges ---------------------------------------------------------


def test_date_text_accepts_dates_and_datetimes() -> None:
    assert store_mod._date_text(date(2024, 1, 2)) == "2024-01-02"  # noqa: SLF001
    assert store_mod._date_text(datetime(2024, 1, 2, 21, 0, tzinfo=UTC)) == "2024-01-02"  # noqa: SLF001


@pytest.mark.parametrize("value", ["2024-01-02", 1700000000, None])
def test_date_text_rejects_non_dates(value: object) -> None:
    with pytest.raises(DataContractError, match="partition date is not a date"):
        store_mod._date_text(value)  # noqa: SLF001


def test_file_entry_for_object_validates_role_and_paths(tmp_path: Path) -> None:
    blob = tmp_path / "b.parquet"
    blob.write_bytes(b"x")
    address = content_address(blob)
    good = dict(
        logical_path="partitions/a/part.parquet",
        object_path="objects/sha256/ab/ab.parquet",
        address=address,
        role="bars",
        source=None,
        symbol=None,
        day=None,
        byte_identical_to_source=True,
    )
    assert file_entry_for_object(**good).role == "bars"
    with pytest.raises(DataContractError, match="unsafe role"):
        file_entry_for_object(**{**good, "role": "../bars"})
    with pytest.raises(DataContractError, match="must stay inside the lake"):
        file_entry_for_object(**{**good, "logical_path": "../part.parquet"})
    with pytest.raises(DataContractError, match="must be YYYY-MM-DD"):
        file_entry_for_object(**{**good, "day": "2024-1-2"})


def test_store_never_leaves_temporary_files_behind(tmp_path: Path) -> None:
    source = tmp_path / "part.parquet"
    source.write_bytes(b"synthetic-bytes")
    put_file(tmp_path, source)
    assert list(tmp_path.rglob("*.tmp")) == []
    assert os.stat(tmp_path / object_relpath(content_address(source).stored_sha256)).st_size
