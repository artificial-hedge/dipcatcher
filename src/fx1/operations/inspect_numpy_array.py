"""Inspect a standalone non-object NPY array without allocating the declared array.

Format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
The bounded header is a Python literal, never executable code. Object/pickle
payloads are rejected. Appended arrays or trailing bytes are not accepted.
"""

from __future__ import annotations

import ast
import hashlib
import math

import numpy as np
from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)


class StructuredField(OutputModel):
    name: str
    dtype: str
    byte_offset: int


class Output(OutputModel):
    format_version: str
    shape: list[int]
    dimension_count: int
    element_count: int
    dtype: str
    dtype_kind: str
    original_descriptor: str
    item_bytes: int
    byte_order: str
    fortran_order: bool
    fields: list[StructuredField]
    payload_offset: int
    payload_bytes: int
    source_sha256: str
    source_bytes: int


def execute(request: Input, context: OperationContext) -> Output:
    """Validate magic, header types, dtype and exact payload size for NPY v1/v2/v3."""
    content = context.read_bytes(request.path, suffixes=(".npy",), max_bytes=8_000_000)
    if len(content) < 10 or content[:6] != b"\x93NUMPY":
        raise ValueError("file is not an NPY array")
    version = tuple(content[6:8])
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError(f"unsupported NPY format version {version}")
    length_bytes = 2 if version == (1, 0) else 4
    header_start = 8 + length_bytes
    if len(content) < header_start:
        raise ValueError("truncated NPY header length")
    header_length = int.from_bytes(content[8:header_start], "little")
    if not 1 <= header_length <= 65_536:
        raise ValueError("NPY header length must be between 1 and 65536 bytes")
    payload_offset = header_start + header_length
    if payload_offset > len(content):
        raise ValueError("truncated NPY header")
    header_bytes = content[header_start:payload_offset]
    if not header_bytes.endswith(b"\n"):
        raise ValueError("NPY header must end with a newline")
    try:
        header_text = header_bytes.decode("utf-8" if version == (3, 0) else "latin1")
        tree = ast.parse(header_text.strip(), mode="eval")
        if not isinstance(tree.body, ast.Dict):
            raise ValueError("NPY header must be a dictionary literal")
        keys = [ast.literal_eval(key) if key is not None else None for key in tree.body.keys]
        if len(keys) != 3 or set(keys) != {"descr", "fortran_order", "shape"}:
            raise ValueError("NPY header requires exactly descr, fortran_order, and shape")
        if sum(1 for _ in ast.walk(tree)) > 20_000:
            raise ValueError("NPY header exceeds 20000 syntax nodes")
        header = ast.literal_eval(tree)
    except (SyntaxError, ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError(f"invalid NPY header: {exc}") from exc
    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or len(shape) > 32
        or any(type(size) is not int or not 0 <= size <= 2**63 - 1 for size in shape)
    ):
        raise ValueError("NPY shape must be a tuple of at most 32 nonnegative 64-bit dimensions")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("NPY fortran_order must be a boolean")
    if not isinstance(header["descr"], (str, list)):
        raise ValueError("NPY dtype descriptor must be a string or structured field list")
    try:
        descriptor = header["descr"]
        dtype = (
            np.dtype(descriptor)
            if isinstance(descriptor, str)
            else np.lib.format.descr_to_dtype(descriptor)
        )
    except (TypeError, ValueError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError(f"unsupported NPY dtype: {exc}") from exc
    if dtype.hasobject:
        raise ValueError("NPY object arrays require pickle and are unsupported")
    element_count = math.prod(shape)
    if element_count > 1_000_000_000:
        raise ValueError("NPY shape exceeds one billion elements")
    payload_bytes = len(content) - payload_offset
    if payload_bytes != element_count * dtype.itemsize:
        raise ValueError("NPY payload length does not match shape and dtype exactly")
    fields: list[StructuredField] = []
    if dtype.names is not None and dtype.fields is not None:
        if len(dtype.names) > 256:
            raise ValueError("NPY structured dtype exceeds 256 fields")
        for name in dtype.names:
            field_dtype, offset = dtype.fields[name][:2]
            fields.append(StructuredField(name=name, dtype=str(field_dtype), byte_offset=offset))
    return Output(
        format_version=f"{version[0]}.{version[1]}",
        shape=list(shape),
        dimension_count=len(shape),
        element_count=element_count,
        dtype=str(dtype),
        dtype_kind=dtype.kind,
        original_descriptor=repr(header["descr"]),
        item_bytes=dtype.itemsize,
        byte_order=dtype.byteorder,
        fortran_order=header["fortran_order"],
        fields=fields,
        payload_offset=payload_offset,
        payload_bytes=payload_bytes,
        source_sha256=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
    )


OPERATION = Operation(
    id="plugins.inspect_numpy_array",
    kind="plugin",
    description=(
        "Inspect standalone NPY v1/v2/v3 array shape, dtype, layout and structured fields; "
        "validate exact payload size without allocating the array or loading pickle."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
