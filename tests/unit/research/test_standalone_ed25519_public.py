"""SYNTHETIC parity and forgery checks for detached Ed25519 auditors."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import ed25519_public as standalone
from scripts import verify_auditor_bundle, verify_corpus_proof, verify_epoch_chain

from quant_fund.utils import ed25519_public as library

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
VERIFIERS = (verify_auditor_bundle, verify_epoch_chain, verify_corpus_proof)
IDENTITY = bytes.fromhex("01" + "00" * 31)
IDENTITY_FORGERY = IDENTITY + bytes(32)


def _body_ast(function: object) -> str:
    node = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(node, ast.FunctionDef)
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        node.body.pop(0)  # Ignore explanatory docstrings, preserve executable code.
    return ast.dump(node, include_attributes=False)


def test_standalone_predicate_matches_library_implementation() -> None:
    """The independent copy must not silently diverge from the shared rule."""
    digest = hashlib.sha256((SCRIPTS / "ed25519_public.py").read_bytes()).hexdigest()
    for verifier in VERIFIERS:
        assert digest == verifier.POINT_CHECKER_SHA256
    for name in ("_P", "_L", "_D", "_SQRT_M1", "_IDENTITY"):
        assert getattr(standalone, name) == getattr(library, name)
    for name in (
        "_decode",
        "_add",
        "_multiply_by_group_order",
        "_is_identity",
        "is_canonical_prime_order_public_key",
    ):
        assert _body_ast(getattr(standalone, name)) == _body_ast(getattr(library, name))


def test_synthetic_point_classes_and_script_signature_entries() -> None:
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    assert cryptography is not None
    private = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    valid = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = b"SYNTHETIC point-class challenge"
    signature = private.sign(message)

    # The Ed25519 base point plus the order-two point has a torsion
    # component: (x, y) + (0, -1) = (-x, -y).
    base_y = int.from_bytes(bytes.fromhex("58" + "66" * 31), "little")
    mixed = ((2**255 - 19 - base_y) | (1 << 255)).to_bytes(32, "little")
    noncanonical = (2**255 - 18).to_bytes(32, "little")
    cases = {"valid": valid, "identity": IDENTITY, "noncanonical": noncanonical, "mixed": mixed}
    for label, key in cases.items():
        expected = label == "valid"
        assert library.is_canonical_prime_order_public_key(key) is expected
        assert standalone.is_canonical_prime_order_public_key(key) is expected

    # The backend alone accepts the identity-point forgery for any message.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    Ed25519PublicKey.from_public_bytes(IDENTITY).verify(IDENTITY_FORGERY, message)
    for verifier in VERIFIERS:
        assert verifier._ed25519_verify(valid.hex(), signature.hex(), message) is True
        assert verifier._ed25519_verify(IDENTITY.hex(), IDENTITY_FORGERY.hex(), message) is False


def test_poisoned_module_cache_cannot_bypass_point_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripts execute hash-pinned source, ignoring imports cached elsewhere."""
    pytest.importorskip("cryptography")
    poisoned = SimpleNamespace(is_canonical_prime_order_public_key=lambda _: True)
    monkeypatch.setitem(sys.modules, "ed25519_public", poisoned)
    monkeypatch.setitem(sys.modules, "scripts.ed25519_public", poisoned)
    for verifier in VERIFIERS:
        assert (
            verifier._ed25519_verify(
                IDENTITY.hex(), IDENTITY_FORGERY.hex(), b"SYNTHETIC cache challenge"
            )
            is False
        )


@pytest.mark.parametrize(
    "script_name", ("verify_auditor_bundle.py", "verify_epoch_chain.py", "verify_corpus_proof.py")
)
def test_detached_copy_uses_only_paired_files(script_name: str, tmp_path: Path) -> None:
    """A copied verifier and checker work with no quant_fund package path."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    key = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = b"SYNTHETIC detached challenge"
    script = tmp_path / script_name
    shutil.copyfile(SCRIPTS / script_name, script)
    shutil.copyfile(SCRIPTS / "ed25519_public.py", tmp_path / "ed25519_public.py")
    code = (
        "import runpy,sys; mod=runpy.run_path(sys.argv[1]); "
        "print(mod['_ed25519_verify'](sys.argv[2],sys.argv[3],sys.argv[4].encode()))"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(script),
            key.hex(),
            private.sign(message).hex(),
            message.decode(),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "True"


@pytest.mark.parametrize("script_name", ("verify_epoch_chain.py", "verify_corpus_proof.py"))
def test_synthetic_identity_forgery_fails_at_cli(script_name: str, tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": "integrity_checkpoint_sig.v1",
                "algorithm": "ed25519",
                "payload": {"heads": {}, "pins": {}},
                "signature": IDENTITY_FORGERY.hex(),
            }
        )
    )
    pubkey = tmp_path / "pubkey"
    pubkey.write_text(IDENTITY.hex())
    if script_name == "verify_epoch_chain.py":
        corpus = tmp_path / "receipts"
        corpus.mkdir()
        target = ["--corpus-dir", str(corpus)]
    else:
        proof = tmp_path / "proof.json"
        proof.write_text("{}")
        target = ["--proof", str(proof)]
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / script_name),
            *target,
            "--checkpoint",
            str(checkpoint),
            "--pubkey",
            str(pubkey),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "checkpoint_signature_invalid" in proc.stdout
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize(
    "script_name", ("verify_auditor_bundle.py", "verify_epoch_chain.py", "verify_corpus_proof.py")
)
def test_tampered_sibling_fails_before_execution(script_name: str, tmp_path: Path) -> None:
    script = tmp_path / script_name
    checker = tmp_path / "ed25519_public.py"
    shutil.copyfile(SCRIPTS / script_name, script)
    shutil.copyfile(SCRIPTS / "ed25519_public.py", checker)
    checker.write_text(checker.read_text() + "\nraise RuntimeError('tampered helper executed')\n")
    if script_name == "verify_auditor_bundle.py":
        args = ["missing-bundle.json", "--offline"]
    elif script_name == "verify_epoch_chain.py":
        corpus = tmp_path / "receipts"
        corpus.mkdir()
        args = ["--corpus-dir", str(corpus)]
    else:
        args = ["--proof", "missing-proof.json", "--pin", "missing-pin.json"]
    proc = subprocess.run(
        [sys.executable, "-I", str(script), *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "ed25519_point_checker_mismatch" in proc.stdout
    assert "tampered helper executed" not in proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
