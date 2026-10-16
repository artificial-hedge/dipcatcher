"""SYNTHETIC local TSA tokens test verifier mechanics, never release evidence."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from quant_fund.research.prereg_timestamp import (
    PreregTimestampError,
    verify_prereg_timestamp,
)
from quant_fund.research.timestamp_anchor import build_tsq


def _openssl() -> Path | None:
    located = shutil.which("openssl")
    if located:
        return Path(located).resolve()
    git_for_windows = Path(r"C:\Program Files\Git\usr\bin\openssl.exe")
    return git_for_windows if git_for_windows.is_file() else None


def _run(executable: Path, *args: str) -> None:
    subprocess.run([str(executable), *args], check=True, capture_output=True, timeout=30)


@pytest.fixture(scope="module")
def synthetic_tsa(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path | str]:
    executable = _openssl()
    if executable is None:
        if os.environ.get("CI", "").lower() in {"true", "1"}:
            pytest.fail("OpenSSL is required to verify preregistration timestamps in CI")
        pytest.skip("OpenSSL is unavailable; production verifier must fail closed")
    root = tmp_path_factory.mktemp("synthetic-prereg-tsa")
    manifest = root / "prereg.json"
    manifest.write_bytes(b'{"schema":"synthetic_prereg.v1","bank_sha256":"abc"}\n')
    ca_key, ca_cert = root / "ca.key", root / "ca.pem"
    tsa_key, tsa_csr, tsa_cert = root / "tsa.key", root / "tsa.csr", root / "tsa.pem"
    _run(
        executable,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(ca_key),
        "-out",
        str(ca_cert),
        "-subj",
        "/CN=synthetic prereg CA",
        "-days",
        "2",
    )
    _run(
        executable,
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(tsa_key),
        "-out",
        str(tsa_csr),
        "-subj",
        "/CN=synthetic prereg TSA",
    )
    extension = root / "tsa.ext"
    extension.write_text(
        "extendedKeyUsage=critical,timeStamping\nkeyUsage=critical,digitalSignature\n",
        encoding="ascii",
    )
    _run(
        executable,
        "x509",
        "-req",
        "-in",
        str(tsa_csr),
        "-CA",
        str(ca_cert),
        "-CAkey",
        str(ca_key),
        "-CAcreateserial",
        "-out",
        str(tsa_cert),
        "-days",
        "2",
        "-extfile",
        str(extension),
    )
    query = root / "query.tsq"
    query.write_bytes(build_tsq(hashlib.sha256(manifest.read_bytes()).hexdigest()))
    token = root / "token.tsr"
    _run(
        executable,
        "ts",
        "-reply",
        "-queryfile",
        str(query),
        "-signer",
        str(tsa_cert),
        "-inkey",
        str(tsa_key),
        "-chain",
        str(ca_cert),
        "-tspolicy",
        "1.2.3.4",
        "-out",
        str(token),
    )
    return {
        "root": root,
        "manifest": manifest,
        "token": token,
        "ca": ca_cert,
        "tsa": tsa_cert,
        "openssl": executable,
        "ca_hash": hashlib.sha256(ca_cert.read_bytes()).hexdigest(),
        "openssl_hash": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }


def _verify(paths: dict[str, Path | str], **overrides: object) -> object:
    args = {
        "manifest_path": paths["manifest"],
        "token_path": paths["token"],
        "ca_cert_path": paths["ca"],
        "tsa_cert_path": paths["tsa"],
        "openssl_executable": paths["openssl"],
        "trusted_ca_sha256": paths["ca_hash"],
        "trusted_openssl_sha256": paths["openssl_hash"],
    }
    args.update(overrides)
    return verify_prereg_timestamp(**args)  # type: ignore[arg-type]


def test_verified_synthetic_token_returns_signed_utc_time(
    synthetic_tsa: dict[str, Path | str],
) -> None:
    before = datetime.now(UTC) - timedelta(minutes=5)
    verified = _verify(synthetic_tsa)
    assert before <= verified.gen_time <= datetime.now(UTC) + timedelta(minutes=5)
    assert (
        verified.manifest_sha256
        == hashlib.sha256(synthetic_tsa["manifest"].read_bytes()).hexdigest()
    )
    assert verified.token_sha256 == hashlib.sha256(synthetic_tsa["token"].read_bytes()).hexdigest()


def test_changed_manifest_and_token_fail_closed(
    synthetic_tsa: dict[str, Path | str], tmp_path: Path
) -> None:
    changed_manifest = tmp_path / "changed.json"
    changed_manifest.write_bytes(synthetic_tsa["manifest"].read_bytes() + b" ")
    with pytest.raises(PreregTimestampError):
        _verify(synthetic_tsa, manifest_path=changed_manifest)
    changed_token = tmp_path / "changed.tsr"
    token = bytearray(synthetic_tsa["token"].read_bytes())
    token[-1] ^= 1
    changed_token.write_bytes(token)
    with pytest.raises(PreregTimestampError):
        _verify(synthetic_tsa, token_path=changed_token)


def test_untrusted_ca_or_executable_refused(synthetic_tsa: dict[str, Path | str]) -> None:
    with pytest.raises(PreregTimestampError, match="CA does not match"):
        _verify(synthetic_tsa, trusted_ca_sha256="0" * 64)
    with pytest.raises(PreregTimestampError, match="OpenSSL executable does not match"):
        _verify(synthetic_tsa, trusted_openssl_sha256="0" * 64)
    with pytest.raises(PreregTimestampError, match="unavailable"):
        _verify(synthetic_tsa, openssl_executable=Path("missing-openssl"))


def test_missing_certificate_refused(synthetic_tsa: dict[str, Path | str], tmp_path: Path) -> None:
    with pytest.raises(PreregTimestampError, match="cannot read timestamp input"):
        _verify(synthetic_tsa, tsa_cert_path=tmp_path / "missing.pem")


def test_wrong_tsa_purpose_refused(synthetic_tsa: dict[str, Path | str], tmp_path: Path) -> None:
    executable = synthetic_tsa["openssl"]
    key, csr, cert = tmp_path / "wrong.key", tmp_path / "wrong.csr", tmp_path / "wrong.pem"
    _run(
        executable,
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key),
        "-out",
        str(csr),
        "-subj",
        "/CN=wrong purpose",
    )
    extension = tmp_path / "wrong.ext"
    extension.write_text("extendedKeyUsage=critical,serverAuth\n", encoding="ascii")
    _run(
        executable,
        "x509",
        "-req",
        "-in",
        str(csr),
        "-CA",
        str(synthetic_tsa["ca"]),
        "-CAkey",
        str(synthetic_tsa["root"] / "ca.key"),
        "-CAcreateserial",
        "-out",
        str(cert),
        "-days",
        "2",
        "-extfile",
        str(extension),
    )
    with pytest.raises(PreregTimestampError, match="chain or purpose"):
        _verify(synthetic_tsa, tsa_cert_path=cert)


def test_other_valid_tsa_under_same_ca_cannot_claim_token(
    synthetic_tsa: dict[str, Path | str], tmp_path: Path
) -> None:
    executable = synthetic_tsa["openssl"]
    key, csr, cert = tmp_path / "other.key", tmp_path / "other.csr", tmp_path / "other.pem"
    _run(
        executable,
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key),
        "-out",
        str(csr),
        "-subj",
        "/CN=other synthetic TSA",
    )
    extension = tmp_path / "other.ext"
    extension.write_text(
        "extendedKeyUsage=critical,timeStamping\nkeyUsage=critical,digitalSignature\n",
        encoding="ascii",
    )
    _run(
        executable,
        "x509",
        "-req",
        "-in",
        str(csr),
        "-CA",
        str(synthetic_tsa["ca"]),
        "-CAkey",
        str(synthetic_tsa["root"] / "ca.key"),
        "-CAcreateserial",
        "-out",
        str(cert),
        "-days",
        "2",
        "-extfile",
        str(extension),
    )
    with pytest.raises(PreregTimestampError, match="chain or purpose"):
        _verify(synthetic_tsa, tsa_cert_path=cert)


def _der_sequence(body: bytes) -> bytes:
    if len(body) < 128:
        length = bytes([len(body)])
    else:
        width = (len(body).bit_length() + 7) // 8
        length = bytes([0x80 | width]) + len(body).to_bytes(width, "big")
    return b"\x30" + length + body


def test_expired_now_tsa_cert_valid_at_signed_generation_time(
    synthetic_tsa: dict[str, Path | str], tmp_path: Path
) -> None:
    """SYNTHETIC re-signed token proves historical validity is evaluated at genTime."""
    executable = synthetic_tsa["openssl"]
    now = datetime.now(UTC).replace(microsecond=0)
    historical = now - timedelta(days=3)
    ca_key_path = synthetic_tsa["root"] / "ca.key"
    tsa_key_path = synthetic_tsa["root"] / "tsa.key"
    ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
    tsa_key = serialization.load_pem_private_key(tsa_key_path.read_bytes(), password=None)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "synthetic historical CA")])
    tsa_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "synthetic prereg TSA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=10))
        .not_valid_after(now + timedelta(days=10))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    tsa_cert = (
        x509.CertificateBuilder()
        .subject_name(tsa_name)
        .issuer_name(ca_name)
        .public_key(tsa_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=5))
        .not_valid_after(now - timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    historical_ca = tmp_path / "historical-ca.pem"
    historical_tsa = tmp_path / "historical-tsa.pem"
    historical_ca.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    historical_tsa.write_bytes(tsa_cert.public_bytes(serialization.Encoding.PEM))

    # The freshly minted fixture token supplies a valid TSTInfo structure.
    # This local mechanics test replaces its UTC time and signs the new bytes
    # with a leaf that was valid then but is expired at the present wall time.
    original_cms = tmp_path / "original.cms"
    original_info = tmp_path / "original-tstinfo.der"
    _run(
        executable,
        "ts",
        "-reply",
        "-in",
        str(synthetic_tsa["token"]),
        "-token_out",
        "-out",
        str(original_cms),
    )
    _run(
        executable,
        "cms",
        "-verify",
        "-noverify",
        "-binary",
        "-inform",
        "DER",
        "-in",
        str(original_cms),
        "-out",
        str(original_info),
    )
    info = original_info.read_bytes()
    times = list(re.finditer(rb"\d{14}Z", info))
    assert len(times) == 1
    old_time = times[0].group()
    new_time = historical.strftime("%Y%m%d%H%M%SZ").encode("ascii")
    historical_info = tmp_path / "historical-tstinfo.der"
    historical_info.write_bytes(info.replace(old_time, new_time, 1))

    cms = tmp_path / "historical.cms"
    _run(
        executable,
        "cms",
        "-sign",
        "-binary",
        "-nodetach",
        "-cades",
        "-econtent_type",
        "1.2.840.113549.1.9.16.1.4",
        "-outform",
        "DER",
        "-in",
        str(historical_info),
        "-signer",
        str(historical_tsa),
        "-inkey",
        str(tsa_key_path),
        "-certfile",
        str(historical_ca),
        "-out",
        str(cms),
    )
    response = tmp_path / "historical.tsr"
    response.write_bytes(_der_sequence(b"\x30\x03\x02\x01\x00" + cms.read_bytes()))
    now_valid = subprocess.run(
        [
            str(executable),
            "verify",
            "-purpose",
            "timestampsign",
            "-CAfile",
            str(historical_ca),
            "-no-CApath",
            "-no-CAstore",
            str(historical_tsa),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert now_valid.returncode != 0  # The same chain is expired today.
    verified = _verify(
        synthetic_tsa,
        token_path=response,
        ca_cert_path=historical_ca,
        tsa_cert_path=historical_tsa,
        trusted_ca_sha256=hashlib.sha256(historical_ca.read_bytes()).hexdigest(),
    )
    assert verified.gen_time == historical
