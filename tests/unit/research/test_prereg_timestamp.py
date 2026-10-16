"""SYNTHETIC local TSA tokens test verifier mechanics, never release evidence."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
