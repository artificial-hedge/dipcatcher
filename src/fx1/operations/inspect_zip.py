"""Inspect ZIP central-directory metadata without extracting or decompressing data.

Reported sizes, CRCs and compression details are declarations in the archive.
They are not a verification of member content or a guarantee of safe extraction.
"""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from collections import Counter
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Member(OutputModel):
    index: int
    name: str
    original_name: str
    name_was_overridden: bool
    directory: bool
    symlink: bool
    encrypted: bool
    compression_method: int
    compressed_bytes: int
    declared_uncompressed_bytes: int
    declared_crc32: str
    expansion_ratio: float | None
    duplicate_name: bool
    path_issues: list[str]
    original_path_issues: list[str]


class Output(OutputModel):
    entry_count: int
    file_count: int
    directory_count: int
    symlink_count: int
    encrypted_entry_count: int
    duplicate_name_count: int
    entries_with_path_issues: int
    path_issue_counts: dict[str, int]
    total_compressed_bytes: int
    total_declared_uncompressed_bytes: int
    members: list[Member]
    offset: int
    has_more: bool
    source_sha256: str
    source_bytes: int


def _path_issues(name: str) -> list[str]:
    issues: list[str] = []
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    if not name:
        issues.append("empty_name")
    if posix.is_absolute() or windows.drive or windows.root:
        issues.append("absolute_or_drive_path")
    if ".." in normalized.split("/"):
        issues.append("parent_traversal")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        issues.append("control_character")
    components = [part for part in normalized.split("/") if part]
    if any(
        PureWindowsPath(part).is_reserved()
        or part.endswith((".", " "))
        or any(character in '<>:"|?*' for character in part)
        for part in components
    ):
        issues.append("windows_ambiguous_name")
    return issues


def execute(request: Input, context: OperationContext) -> Output:
    """Summarize the full bounded directory and return one page of member metadata."""
    content = context.read_bytes(request.path, suffixes=(".zip", ".npz"), max_bytes=8_000_000)
    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, NotImplementedError, UnicodeError) as exc:
        raise ValueError(f"invalid or unsupported ZIP central directory: {exc}") from exc
    if len(entries) > 10_000:
        raise ValueError("ZIP directory exceeds 10000 entries")
    # Unicode Path extra records can override the originally decoded filename.
    # The effective name is what ZipFile uses to locate or extract the member.
    names = Counter(info.filename for info in entries)
    members: list[Member] = []
    directory_count = symlinks = encrypted = path_problem_count = 0
    compressed_total = uncompressed_total = 0
    issue_counts: Counter[str] = Counter()
    for index, info in enumerate(entries):
        if len(info.orig_filename) > 4096 or len(info.filename) > 4096:
            raise ValueError("ZIP member names cannot exceed 4096 characters")
        directory = info.is_dir()
        symlink = info.create_system == 3 and stat.S_ISLNK(info.external_attr >> 16)
        is_encrypted = bool(info.flag_bits & 1)
        issues = _path_issues(info.filename)
        original_issues = _path_issues(info.orig_filename)
        all_issues = set(issues) | set(original_issues)
        directory_count += directory
        symlinks += symlink
        encrypted += is_encrypted
        path_problem_count += bool(all_issues)
        issue_counts.update(all_issues)
        compressed_total += info.compress_size
        uncompressed_total += info.file_size
        if request.offset <= index < request.offset + request.limit:
            members.append(
                Member(
                    index=index,
                    name=info.filename,
                    original_name=info.orig_filename,
                    name_was_overridden=info.filename != info.orig_filename,
                    directory=directory,
                    symlink=symlink,
                    encrypted=is_encrypted,
                    compression_method=info.compress_type,
                    compressed_bytes=info.compress_size,
                    declared_uncompressed_bytes=info.file_size,
                    declared_crc32=f"{info.CRC:08x}",
                    expansion_ratio=(
                        info.file_size / info.compress_size if info.compress_size else None
                    ),
                    duplicate_name=names[info.filename] > 1,
                    path_issues=issues,
                    original_path_issues=original_issues,
                )
            )
    return Output(
        entry_count=len(entries),
        file_count=len(entries) - directory_count,
        directory_count=directory_count,
        symlink_count=symlinks,
        encrypted_entry_count=encrypted,
        duplicate_name_count=sum(count > 1 for count in names.values()),
        entries_with_path_issues=path_problem_count,
        path_issue_counts=dict(sorted(issue_counts.items())),
        total_compressed_bytes=compressed_total,
        total_declared_uncompressed_bytes=uncompressed_total,
        members=members,
        offset=request.offset,
        has_more=request.offset + len(members) < len(entries),
        source_sha256=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
    )


OPERATION = Operation(
    id="plugins.inspect_zip",
    kind="plugin",
    description=(
        "Inspect ZIP/NPZ directory metadata, declared sizes, duplicate names, path hazards, "
        "symlinks and encryption with pagination. Does not extract, decompress or verify members."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
