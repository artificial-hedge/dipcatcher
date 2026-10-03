"""Compare a bounded workspace data artifact with an explicitly supplied digest.

Byte identity establishes neither the source's trustworthiness nor research
eligibility. The result describes only the exact bytes read in this invocation.
"""

import hmac

from pydantic import Field, field_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_SUFFIXES = (
    ".json",
    ".jsonl",
    ".ndjson",
    ".csv",
    ".tsv",
    ".parquet",
    ".arrow",
    ".ipc",
    ".feather",
    ".txt",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".npy",
    ".zip",
    ".gz",
)


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    expected_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$", min_length=64, max_length=64)
    expected_bytes: int | None = Field(default=None, strict=True, ge=0, le=1_000_000_000)
    max_bytes: int = Field(default=64_000_000, strict=True, ge=1, le=1_000_000_000)

    @field_validator("expected_sha256")
    @classmethod
    def normalize_digest(cls, value: str) -> str:
        return value.lower()


class Output(OutputModel):
    path: str
    expected_sha256: str
    actual_sha256: str
    digest_matches: bool
    actual_bytes: int
    expected_bytes: int | None
    size_matches: bool | None
    matches: bool


def execute(request: Input, context: OperationContext) -> Output:
    """Hash the same bounded bytes used for the reported size comparison."""
    with context.open_binary(
        request.path, suffixes=_SUFFIXES, max_bytes=request.max_bytes
    ) as stream:
        while stream.read(65_536):
            pass
        actual, actual_bytes = stream.source_sha256, stream.bytes_read
    digest_matches = hmac.compare_digest(actual, request.expected_sha256)
    size_matches = (
        None if request.expected_bytes is None else actual_bytes == request.expected_bytes
    )
    return Output(
        path=request.path,
        expected_sha256=request.expected_sha256,
        actual_sha256=actual,
        digest_matches=digest_matches,
        actual_bytes=actual_bytes,
        expected_bytes=request.expected_bytes,
        size_matches=size_matches,
        matches=digest_matches and size_matches is not False,
    )


OPERATION = Operation(
    id="plugins.verify_file_hash",
    kind="plugin",
    description=(
        "Stream a workspace data file against an expected SHA-256 and optional byte count "
        "under a caller-selected budget (64 MB default, 1 GB maximum). A match identifies "
        "bytes only and does not establish research eligibility."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
    version="1.1.0",
)
