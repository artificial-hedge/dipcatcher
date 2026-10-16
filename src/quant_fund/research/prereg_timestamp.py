"""Strict RFC 3161 timestamp verification for preregistered evidence.

The caller supplies independently approved hashes for the TSA CA and the
OpenSSL executable. A timestamp is returned only after OpenSSL validates the
CMS signature against the supplied TSA leaf, its timestamp-signing purpose,
its trusted chain, and the exact SHA-256
message imprint. The signed TSTInfo is then parsed for its UTC ``genTime``.
This proves that the manifest bytes existed by that time; it cannot prove
that an evaluator kept a bank secret or ran a model after that time.
"""

from __future__ import annotations

import hashlib
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_GENERALIZED_TIME = re.compile(rb"(\d{14})(?:\.(\d{1,6}))?Z\Z")
_SIGNED_DATA_OID = bytes.fromhex("2a864886f70d010702")
_TST_INFO_OID = bytes.fromhex("2a864886f70d0109100104")
_SHA256_OID = bytes.fromhex("608648016503040201")


class PreregTimestampError(ValueError):
    """The preregistration timestamp has not been independently verified."""


@dataclass(frozen=True)
class VerifiedPreregTimestamp:
    gen_time: datetime
    manifest_sha256: str
    token_sha256: str


def _read_regular(path: Path, *, max_bytes: int) -> bytes:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise PreregTimestampError(f"missing, nonregular, or oversized input: {path}")
        data = path.read_bytes()
    except OSError as exc:
        raise PreregTimestampError(f"cannot read timestamp input: {path}") from exc
    if len(data) > max_bytes:
        raise PreregTimestampError(f"oversized timestamp input: {path}")
    return data


def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int]:
    if offset + 2 > len(data):
        raise PreregTimestampError("truncated timestamp DER")
    tag = data[offset]
    if tag & 0x1F == 0x1F:
        raise PreregTimestampError("unsupported timestamp DER tag")
    first = data[offset + 1]
    pos = offset + 2
    if first & 0x80:
        nbytes = first & 0x7F
        if nbytes == 0 or nbytes > 4 or pos + nbytes > len(data):
            raise PreregTimestampError("invalid timestamp DER length")
        length_bytes = data[pos : pos + nbytes]
        length = int.from_bytes(length_bytes, "big")
        if length < 128 or length_bytes[0] == 0:
            raise PreregTimestampError("noncanonical timestamp DER length")
        pos += nbytes
    else:
        length = first
    end = pos + length
    if end > len(data):
        raise PreregTimestampError("truncated timestamp DER value")
    return tag, data[pos:end], end


def _one(data: bytes, expected_tag: int) -> bytes:
    tag, body, end = _read_tlv(data, 0)
    if tag != expected_tag or end != len(data):
        raise PreregTimestampError("unexpected timestamp DER structure")
    return body


def _children(body: bytes) -> list[tuple[int, bytes]]:
    result: list[tuple[int, bytes]] = []
    pos = 0
    while pos < len(body):
        tag, value, pos = _read_tlv(body, pos)
        result.append((tag, value))
    return result


def _signed_token_der(token: bytes) -> bytes:
    response = _one(token, 0x30)
    status_tag, _, status_end = _read_tlv(response, 0)
    token_tag, _, token_end = _read_tlv(response, status_end)
    if status_tag != 0x30 or token_tag != 0x30 or token_end != len(response):
        raise PreregTimestampError("timestamp response has no signed token")
    return response[status_end:token_end]


def _signed_tst_info(token: bytes) -> bytes:
    content = _children(_one(_signed_token_der(token), 0x30))
    if len(content) != 2 or content[0] != (0x06, _SIGNED_DATA_OID) or content[1][0] != 0xA0:
        raise PreregTimestampError("timestamp token is not CMS SignedData")
    signed_data = _children(_one(content[1][1], 0x30))
    if (
        len(signed_data) < 3
        or signed_data[0][0] != 0x02
        or signed_data[1][0] != 0x31
        or signed_data[2][0] != 0x30
    ):
        raise PreregTimestampError("malformed timestamp SignedData")
    encap = _children(signed_data[2][1])
    if len(encap) != 2 or encap[0] != (0x06, _TST_INFO_OID) or encap[1][0] != 0xA0:
        raise PreregTimestampError("timestamp token lacks TSTInfo")
    return _one(encap[1][1], 0x04)


def _signed_time_and_imprint(token: bytes) -> tuple[datetime, bytes]:
    fields = _children(_one(_signed_tst_info(token), 0x30))
    if len(fields) < 5 or [tag for tag, _ in fields[:5]] != [0x02, 0x06, 0x30, 0x02, 0x18]:
        raise PreregTimestampError("malformed timestamp TSTInfo")
    imprint_fields = _children(fields[2][1])
    if len(imprint_fields) != 2 or imprint_fields[0][0] != 0x30 or imprint_fields[1][0] != 0x04:
        raise PreregTimestampError("malformed timestamp message imprint")
    algorithm = _children(imprint_fields[0][1])
    if not algorithm or algorithm[0] != (0x06, _SHA256_OID):
        raise PreregTimestampError("timestamp imprint must use SHA-256")
    if len(algorithm) > 2 or (len(algorithm) == 2 and algorithm[1] != (0x05, b"")):
        raise PreregTimestampError("malformed timestamp SHA-256 algorithm")
    imprint = imprint_fields[1][1]
    if len(imprint) != 32:
        raise PreregTimestampError("timestamp SHA-256 imprint has wrong length")
    match = _GENERALIZED_TIME.fullmatch(fields[4][1])
    if match is None:
        raise PreregTimestampError("timestamp genTime must be UTC GeneralizedTime")
    try:
        gen_time = datetime.strptime(match[1].decode("ascii"), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if match[2] is not None:
            gen_time = gen_time.replace(microsecond=int(match[2].ljust(6, b"0")))
    except ValueError as exc:
        raise PreregTimestampError("invalid timestamp genTime") from exc
    return gen_time, imprint


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_HEX.fullmatch(value) is None:
        raise PreregTimestampError(f"{label} must be lowercase SHA-256 hex")


def _run_openssl(executable: Path, *args: str) -> None:
    try:
        result = subprocess.run(
            [str(executable), *args],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreregTimestampError("OpenSSL timestamp verification unavailable") from exc
    if result.returncode != 0:
        raise PreregTimestampError("OpenSSL timestamp chain or purpose verification failed")


def verify_prereg_timestamp(
    manifest_path: Path,
    token_path: Path,
    *,
    ca_cert_path: Path,
    tsa_cert_path: Path,
    openssl_executable: Path,
    trusted_ca_sha256: str,
    trusted_openssl_sha256: str,
) -> VerifiedPreregTimestamp:
    """Verify an exact prereg manifest against an independently trusted TSA.

    The CA and verifier hashes must come from a trust registry established
    before the evaluation. This helper does not enroll either trust root.
    The pinned executable must remain immutable between its hash check and
    execution; use an immutable deployment image to enforce that assumption.
    """
    _require_sha256(trusted_ca_sha256, "trusted_ca_sha256")
    _require_sha256(trusted_openssl_sha256, "trusted_openssl_sha256")
    manifest = _read_regular(manifest_path, max_bytes=1024 * 1024)
    token = _read_regular(token_path, max_bytes=64 * 1024)
    ca_cert = _read_regular(ca_cert_path, max_bytes=64 * 1024)
    tsa_cert = _read_regular(tsa_cert_path, max_bytes=64 * 1024)
    if hashlib.sha256(ca_cert).hexdigest() != trusted_ca_sha256:
        raise PreregTimestampError("TSA CA does not match the trusted fingerprint")
    try:
        executable = openssl_executable.resolve(strict=True)
        if (
            not executable.is_file()
            or hashlib.sha256(executable.read_bytes()).hexdigest() != trusted_openssl_sha256
        ):
            raise PreregTimestampError("OpenSSL executable does not match the trusted fingerprint")
    except OSError as exc:
        raise PreregTimestampError("trusted OpenSSL executable is unavailable") from exc

    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    cms_der = _signed_token_der(token)
    signed_tst_info = _signed_tst_info(token)
    with tempfile.TemporaryDirectory(prefix="prereg-timestamp-") as scratch:
        temp = Path(scratch)
        token_copy = temp / "token.tsr"
        cms_copy = temp / "token.cms"
        verified_content = temp / "verified-tstinfo.der"
        ca_copy = temp / "ca.pem"
        tsa_copy = temp / "tsa.pem"
        token_copy.write_bytes(token)
        cms_copy.write_bytes(cms_der)
        ca_copy.write_bytes(ca_cert)
        tsa_copy.write_bytes(tsa_cert)
        _run_openssl(
            executable,
            "ts",
            "-verify",
            "-digest",
            manifest_sha256,
            "-in",
            str(token_copy),
            "-CAfile",
            str(ca_copy),
            "-untrusted",
            str(tsa_copy),
        )
        _run_openssl(
            executable,
            "verify",
            "-purpose",
            "timestampsign",
            "-CAfile",
            str(ca_copy),
            "-no-CApath",
            "-no-CAstore",
            str(tsa_copy),
        )
        _run_openssl(
            executable,
            "cms",
            "-verify",
            "-inform",
            "DER",
            "-in",
            str(cms_copy),
            "-nointern",
            "-certfile",
            str(tsa_copy),
            "-CAfile",
            str(ca_copy),
            "-no-CApath",
            "-no-CAstore",
            "-purpose",
            "timestampsign",
            "-binary",
            "-out",
            str(verified_content),
        )
        if verified_content.read_bytes() != signed_tst_info:
            raise PreregTimestampError("CMS signature did not cover the parsed timestamp TSTInfo")

    gen_time, imprint = _signed_time_and_imprint(token)
    if imprint != bytes.fromhex(manifest_sha256):
        raise PreregTimestampError("timestamp imprint differs from exact prereg manifest")
    return VerifiedPreregTimestamp(
        gen_time=gen_time,
        manifest_sha256=manifest_sha256,
        token_sha256=hashlib.sha256(token).hexdigest(),
    )
