"""Inspect strict, uncompressed USTAR/V7 archives without extracting anything.

The parser checks each 512-byte header, unsigned checksum, bounded octal fields,
UTF-8 names and the exact declared payload extent. It accepts regular files,
links, directories, devices, FIFOs and contiguous files. PAX, GNU (including
sparse/long-name), compression and unknown extension formats are rejected.
Member padding must be zero. Two zero end blocks are required; remaining bytes
must be zero block padding. Concatenated archives and trailing junk are rejected.

All members are scanned before pagination. Limits are 32 MB source, 1000 members
and 16 MB per payload. Paths and link targets are inspected as declarations;
reported path concerns are observations, never extraction authorization. Source
hash includes headers and padding; payload hashes exclude block padding.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import PureWindowsPath
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_TYPES = {
    b"\0": "regular",
    b"0": "regular",
    b"1": "hard_link",
    b"2": "symbolic_link",
    b"3": "character_device",
    b"4": "block_device",
    b"5": "directory",
    b"6": "fifo",
    b"7": "contiguous",
}


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=1000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Member(OutputModel):
    index: int
    header_offset: int
    format: Literal["ustar", "v7"]
    name: str
    kind: str
    size: int
    mode: int
    uid: int
    gid: int
    mtime_unix_seconds: int
    link_target: str | None
    device_major: int | None
    device_minor: int | None
    path_issues: list[str]
    link_target_issues: list[str]
    duplicate_name: bool
    header_sha256: str
    payload_sha256: str | None


class Output(OutputModel):
    member_count: int
    members: list[Member]
    offset: int
    has_more: bool
    type_counts: dict[str, int]
    total_payload_bytes: int
    duplicate_name_count: int
    members_with_path_issues: int
    end_padding_bytes: int
    archive_structure_validated: Literal[True] = True
    extraction_performed: Literal[False] = False
    source_bytes: int
    source_sha256: str


def _read(stream: io.RawIOBase, size: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < size:
        chunk = stream.read(size - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _text(value: bytes) -> str:
    text, separator, padding = value.partition(b"\0")
    if separator and any(padding):
        raise ValueError("tar text fields must have zero padding after their terminator")
    return text.decode("utf-8")


def _octal(value: bytes, name: str, *, blank: bool = False) -> int:
    if value and value[0] & 0x80:
        raise ValueError("GNU base-256 numeric fields are unsupported")
    text = value.strip(b" \0")
    if not text and blank:
        return 0
    if not text or any(character not in b"01234567" for character in text):
        raise ValueError(f"tar {name} must be a bounded nonnegative octal field")
    return int(text, 8)


def _path_issues(value: str, *, directory: bool = False) -> list[str]:
    issues: list[str] = []
    path = value[:-1] if directory and value.endswith("/") else value
    components = path.split("/")
    if not path:
        issues.append("empty_path")
    if path.startswith("/") or PureWindowsPath(path).is_absolute():
        issues.append("absolute_path")
    if PureWindowsPath(path).drive or ":" in path:
        issues.append("windows_drive_or_stream")
    if "\\" in path:
        issues.append("backslash_separator")
    if ".." in components:
        issues.append("parent_component")
    if "." in components:
        issues.append("dot_component")
    if "" in components and path:
        issues.append("empty_component")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        issues.append("control_character")
    if any(PureWindowsPath(component).is_reserved() for component in components):
        issues.append("windows_reserved_component")
    if any(component.endswith((".", " ")) for component in components):
        issues.append("trailing_dot_or_space")
    return issues


def execute(request: Input, context: OperationContext) -> Output:
    members: list[Member] = []
    names: set[str] = set()
    type_counts: dict[str, int] = {}
    count = payload_bytes = duplicates = path_concerns = 0
    end_padding = 0
    try:
        with context.open_binary(request.path, suffixes=(".tar",), max_bytes=32_000_000) as stream:
            while True:
                header_offset = stream.bytes_read
                header = _read(stream, 512)
                if len(header) != 512:
                    raise ValueError("tar lacks a complete header or two zero end blocks")
                if not any(header):
                    if _read(stream, 512) != b"\0" * 512:
                        raise ValueError("tar requires two consecutive zero end blocks")
                    end_padding = 1024
                    while chunk := stream.read(65_536):
                        if any(chunk):
                            raise ValueError(
                                "tar has nonzero trailing bytes or a concatenated archive"
                            )
                        end_padding += len(chunk)
                    if end_padding % 512:
                        raise ValueError("tar end padding must consist of complete 512-byte blocks")
                    break
                if count >= 1000:
                    raise ValueError("tar exceeds 1000 members")
                if header[:2] == b"\x1f\x8b" or header[:3] == b"BZh" or header[:6] == b"\xfd7zXZ\0":
                    raise ValueError("compressed tar archives are unsupported")
                kind_byte = header[156:157]
                if kind_byte not in _TYPES:
                    raise ValueError(
                        "tar extension/unknown member types, including sparse/PAX/GNU, are unsupported"
                    )
                stored_checksum = _octal(header[148:156], "checksum")
                if stored_checksum != sum(header[:148]) + 8 * 32 + sum(header[156:]):
                    raise ValueError("tar header unsigned checksum mismatch")
                if header[257:263] == b"ustar\0" and header[263:265] == b"00":
                    archive_format: Literal["ustar", "v7"] = "ustar"
                    if any(header[500:512]):
                        raise ValueError("USTAR reserved header bytes must be zero")
                    prefix = _text(header[345:500])
                    name = _text(header[:100])
                    name = f"{prefix}/{name}" if prefix else name
                    _text(header[265:297])
                    _text(header[297:329])
                    major = _octal(header[329:337], "device major", blank=True)
                    minor = _octal(header[337:345], "device minor", blank=True)
                elif not any(header[257:512]):
                    archive_format = "v7"
                    name = _text(header[:100])
                    major = minor = 0
                else:
                    raise ValueError("only plain USTAR and V7 tar headers are supported")
                kind = _TYPES[kind_byte]
                if kind_byte == b"\0" and name.endswith("/"):
                    kind = "directory"
                size = _octal(header[124:136], "size")
                if size > 16_000_000:
                    raise ValueError("tar member payload exceeds 16 MB")
                if kind not in ("regular", "contiguous") and size:
                    raise ValueError("non-file tar members must declare zero payload bytes")
                mode = _octal(header[100:108], "mode")
                uid = _octal(header[108:116], "uid")
                gid = _octal(header[116:124], "gid")
                mtime = _octal(header[136:148], "mtime")
                target = _text(header[157:257])
                if target and kind not in ("hard_link", "symbolic_link"):
                    raise ValueError("tar link targets are supported only for link members")
                digest = hashlib.sha256()
                remaining = size
                while remaining:
                    chunk = _read(stream, min(65_536, remaining))
                    if not chunk:
                        raise ValueError("truncated tar member payload")
                    remaining -= len(chunk)
                    digest.update(chunk)
                padding = (-size) % 512
                if _read(stream, padding) != b"\0" * padding:
                    raise ValueError("tar member padding is truncated or nonzero")
                issues = _path_issues(name, directory=kind == "directory")
                link_issues = _path_issues(target) if kind in ("hard_link", "symbolic_link") else []
                duplicate = name in names
                duplicates += int(duplicate)
                path_concerns += int(bool(issues or link_issues))
                names.add(name)
                type_counts[kind] = type_counts.get(kind, 0) + 1
                payload_bytes += size
                if request.offset <= count < request.offset + request.limit:
                    members.append(
                        Member(
                            index=count,
                            header_offset=header_offset,
                            format=archive_format,
                            name=name,
                            kind=kind,
                            size=size,
                            mode=mode,
                            uid=uid,
                            gid=gid,
                            mtime_unix_seconds=mtime,
                            link_target=target if kind in ("hard_link", "symbolic_link") else None,
                            device_major=major
                            if kind in ("character_device", "block_device")
                            else None,
                            device_minor=minor
                            if kind in ("character_device", "block_device")
                            else None,
                            path_issues=issues,
                            link_target_issues=link_issues,
                            duplicate_name=duplicate,
                            header_sha256=hashlib.sha256(header).hexdigest(),
                            payload_sha256=digest.hexdigest()
                            if kind in ("regular", "contiguous")
                            else None,
                        )
                    )
                count += 1
            source_bytes = stream.bytes_read
            source_hash = stream.source_sha256
    except UnicodeError as exc:
        raise ValueError("tar metadata must be strict UTF-8") from exc
    return Output(
        member_count=count,
        members=members,
        offset=request.offset,
        has_more=request.offset + len(members) < count,
        type_counts=type_counts,
        total_payload_bytes=payload_bytes,
        duplicate_name_count=duplicates,
        members_with_path_issues=path_concerns,
        end_padding_bytes=end_padding,
        source_bytes=source_bytes,
        source_sha256=source_hash,
    )


OPERATION = Operation(
    id="plugins.inspect_tar",
    kind="plugin",
    description="Inspect and hash all members of a bounded plain USTAR/V7 archive without extraction; reject compression, sparse/extensions, bad checksums, truncation and trailing junk, and report path concerns.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
