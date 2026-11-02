"""Coverage for ``utils/reproducibility.py``'s content addressing.

``content_address`` is the public seam used by lineage/registry receipts. Its
Git LFS branch (pointer file vs. smudged payload) and every malformed-pointer
rejection are exercised here with synthetic bytes — no git binary required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quant_fund.utils.reproducibility import (
    _parse_lfs_pointer,
    _parse_pointer_batch,
    content_address,
)

OID = "a" * 64


def _pointer(oid: str = OID, size: int = 4096) -> bytes:
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + oid.encode("ascii") + b"\n"
        b"size " + str(size).encode("ascii") + b"\n"
    )


def test_content_address_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        content_address(tmp_path / "absent.parquet")


def test_content_address_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        content_address(tmp_path)


def test_content_address_hashes_an_ordinary_blob(tmp_path: Path) -> None:
    payload = b"SYNTHETIC bars\x00\x01"
    path = tmp_path / "bars.bin"
    path.write_bytes(payload)

    address = content_address(path)

    expected = hashlib.sha256(payload).hexdigest()
    assert address.kind == "blob"
    assert address.content_sha256 == expected
    assert address.stored_sha256 == expected
    assert address.stored_size == len(payload)
    assert address.declared_size == len(payload)


def test_content_address_resolves_an_lfs_pointer_to_its_object_id(tmp_path: Path) -> None:
    raw = _pointer(size=123456)
    path = tmp_path / "gold.parquet"
    path.write_bytes(raw)

    address = content_address(path)

    assert address.kind == "lfs_pointer"
    assert address.content_sha256 == OID
    assert address.stored_sha256 == hashlib.sha256(raw).hexdigest()
    assert address.stored_size == len(raw)
    assert address.declared_size == 123456


def test_content_address_accepts_crlf_pointer_files(tmp_path: Path) -> None:
    path = tmp_path / "gold.parquet"
    path.write_bytes(_pointer().replace(b"\n", b"\r\n"))

    address = content_address(path)

    assert address.kind == "lfs_pointer"
    assert address.content_sha256 == OID


@pytest.mark.parametrize(
    "blob",
    [
        pytest.param(b"not a pointer at all\n", id="bad-header"),
        pytest.param(_pointer(oid="A" * 64), id="uppercase-oid"),
        pytest.param(_pointer(oid="a" * 63), id="short-oid"),
        pytest.param(_pointer(oid="z" * 64), id="non-hex-oid"),
        pytest.param(_pointer().replace(b"size 4096", b"size four"), id="non-numeric-size"),
        pytest.param(_pointer().replace(b"\nsize 4096", b""), id="no-size"),
        pytest.param(_pointer().replace(b"oid sha256:" + OID.encode(), b""), id="no-oid"),
        pytest.param(_pointer() + b"\x00", id="nul-byte"),
        pytest.param(_pointer() + b"x" * 2048, id="oversize"),
        pytest.param(_pointer() + b"\xff", id="non-ascii"),
    ],
)
def test_malformed_pointers_are_not_resolved(tmp_path: Path, blob: bytes) -> None:
    assert _parse_lfs_pointer(blob) is None

    path = tmp_path / "not-a-pointer.bin"
    path.write_bytes(blob)
    address = content_address(path)
    assert address.kind == "blob"
    assert address.content_sha256 == address.stored_sha256


def test_pointer_parser_accepts_blank_lines_between_fields() -> None:
    blob = _pointer().replace(b"\noid", b"\n\noid")
    assert _parse_lfs_pointer(blob) == (OID, 4096)


def _batch_entry(oid: str, payload: bytes) -> bytes:
    header = f"{oid} blob {len(payload)}\n".encode("ascii")
    return header + payload + b"\n"


def test_parse_pointer_batch_reads_each_path_in_order() -> None:
    first = _pointer(size=1)
    second = _pointer(oid="b" * 64, size=2)
    stdout = _batch_entry("1" * 40, first) + _batch_entry("2" * 40, second)

    parsed = _parse_pointer_batch(stdout, [b"one.bin", b"two.bin"])

    assert parsed == {b"one.bin": (OID, 1), b"two.bin": ("b" * 64, 2)}


def test_parse_pointer_batch_skips_missing_and_non_blob_entries() -> None:
    stdout = b"<missing>\n" + b"1" * 40 + b" tree 3\nabc"

    assert _parse_pointer_batch(stdout, [b"genuinely-missing.bin", b"dir"]) == {}


def test_parse_pointer_batch_with_no_headers_is_empty() -> None:
    assert _parse_pointer_batch(b"", [b"one.bin"]) == {}
