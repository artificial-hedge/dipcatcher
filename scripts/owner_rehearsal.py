#!/usr/bin/env python3
"""Rehearse OWNER_STEPS.md in a throwaway copy of this checkout.

The copy lives under /tmp. The script refuses a destination that is the
real checkout or nested inside it, and it checks the source tree afterwards.
It does not push. Keys, signatures, epochs, and weights from the rehearsal
stay in the copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from quant_fund.research.gate_signatures import (
    generate_keypair,
    init_quorum,
    key_id,
    quorum_registry_errors,
    registry_file_bytes,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

_MARKER = ".owner-rehearsal"
_SENTINELS = (
    "gate_pins.sig",
    "quality/gate_signing.pub",
    "quality/gate_quorum.json",
    "quality/checkpoint.json",
    "quality/epoch_heads.json",
    "quality/crown_jewels.json",
)
_CARD = {
    "version": "fx-1-rehearsal",
    "base_model": "fx-1",
    "known_limits": ["SYNTHETIC rehearsal weights for the owner ceremony"],
}


class RehearsalRefusal(RuntimeError):
    """The rehearsal would write the real checkout."""


def assert_throwaway(source: Path, dest: Path) -> None:
    """Refuse a destination that is, contains, or sits inside the source."""
    src = source.resolve()
    dst = dest.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise RehearsalRefusal("owner-rehearsal refuses to write the real checkout")


def require_copy(source: Path, dest: Path) -> None:
    """Refuse unless dest is a marked throwaway outside the source tree."""
    assert_throwaway(source, dest)
    marker = dest / _MARKER
    if not marker.is_file():
        raise RehearsalRefusal("owner-rehearsal refuses a tree without the throwaway marker")


def unlink_under(root: Path, path: Path) -> None:
    """Unlink a regular file that resolves inside root."""
    if path.is_symlink():
        raise RehearsalRefusal("refusing to unlink a symlink")
    resolved = path.resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise RehearsalRefusal("refusing to unlink outside the throwaway copy")
    path.unlink()


def public_hex(seed_hex: str) -> str:
    """Ed25519 public key hex for a 32-byte seed. The seed is not printed."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex.strip()))
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def retarget_data_manifest(source: Path, dest: Path) -> bool:
    """Point a copied manifest at dest and reseal it.

    Ingest records absolute artifact paths. A throwaway copy fails doctor
    on ``inside_root`` until those paths name the copy. The owner's checkout
    is already the path in the manifest, so OWNER_STEPS does not do this.
    """
    require_copy(source, dest)
    manifest = dest / "data" / "metadata" / "data_manifest.json"
    if not manifest.is_file():
        return False
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = blob.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    prefix = str(source.resolve()) + "/"
    moved = _retarget_artifact_paths(artifacts, prefix, dest.resolve())
    if not moved:
        return False
    unsigned = {key: value for key, value in blob.items() if key != "receipt_sha256"}
    unsigned["receipt_sha256"] = hash_bytes(canonical_json_bytes(unsigned))
    manifest.write_text(
        json.dumps(unsigned, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def _retarget_artifact_paths(artifacts: dict[str, object], prefix: str, dest: Path) -> bool:
    moved = False
    for item in artifacts.values():
        if not isinstance(item, dict):
            continue
        raw = item.get("path")
        if not isinstance(raw, str) or not raw.startswith(prefix):
            continue
        item["path"] = str(dest / raw[len(prefix) :])
        moved = True
    return moved


def receipt_drift(root: Path) -> list[tuple[str, str]]:
    """Receipt members whose live bytes differ from the pinned head epoch.

    Receipts are append-only. A later stamp records a digest change as
    ``member_mutated`` and the chain fails. The list is ``(name, stamped
    sha256)`` for missing files and for files whose bytes moved.
    """
    allowed = _allowed_removals(root)
    drifted: list[tuple[str, str]] = []
    for name, digest in _head_receipt_members(root).items():
        path = _receipt_member_path(root, name)
        if path.is_symlink():
            drifted.append((name, digest))
            continue
        if not path.is_file():
            if allowed.get(name) != digest:
                drifted.append((name, digest))
            continue
        if hash_bytes(path.read_bytes()) != digest:
            drifted.append((name, digest))
    return drifted


def historical_receipt_bytes(root: Path, name: str, digest: str) -> bytes | None:
    """Bytes of ``receipts/<name>`` from git history whose sha256 is ``digest``."""
    _receipt_member_path(root, name)
    if not _sha256_hex(digest):
        return None
    rel = f"receipts/{name}"
    listed = subprocess.run(
        ["git", "-C", str(root), "rev-list", "HEAD", "--", rel],
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        return None
    for rev in listed.stdout.split():
        if not _git_rev(rev):
            continue
        shown = subprocess.run(
            ["git", "-C", str(root), "show", f"{rev.decode('ascii')}:{rel}"],
            capture_output=True,
            check=False,
        )
        if shown.returncode == 0 and hash_bytes(shown.stdout) == digest:
            return bytes(shown.stdout)
    return None


def restore_stamped_receipts(root: Path) -> tuple[list[str], list[str]]:
    """Put drifted receipts back to the digest the head epoch already names.

    The bytes come from this checkout's git history. A member with no such
    blob is listed as unrestored and left untouched. Symlinks and paths that
    leave ``receipts/`` are refused. Epoch receipt files are not rewritten.
    """
    restored: list[str] = []
    missing: list[str] = []
    for name, digest in receipt_drift(root):
        if _epoch_receipt_name(name):
            missing.append(name)
            continue
        blob = historical_receipt_bytes(root, name, digest)
        if blob is None:
            missing.append(name)
            continue
        path = _receipt_member_path(root, name)
        _replace_receipt(path, blob)
        if path.is_symlink() or hash_bytes(path.read_bytes()) != digest:
            raise RehearsalRefusal(f"restore did not land {name}")
        restored.append(name)
    return restored, missing


def _allowed_removals(root: Path) -> dict[str, str]:
    """Name to sha256 pins for receipts the chain already allowed to leave."""
    path = root / "quality" / "epoch_allowed_removals.json"
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RehearsalRefusal("allowed removals pin is unreadable") from exc
    if not isinstance(raw, dict):
        raise RehearsalRefusal("allowed removals pin is malformed")
    return {
        key: value
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, str) and _sha256_hex(value)
    }


def _head_receipt_members(root: Path) -> dict[str, str]:
    pin_path = root / "quality" / "epoch_heads.json"
    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RehearsalRefusal("epoch heads pin is unreadable") from exc
    if not isinstance(pin, dict):
        raise RehearsalRefusal("epoch heads pin is not an object")
    heads = pin.get("heads")
    if not isinstance(heads, dict):
        raise RehearsalRefusal("epoch heads pin has no heads map")
    entry = heads.get("receipts/*.json")
    if not isinstance(entry, dict):
        raise RehearsalRefusal("epoch heads pin has no receipts chain")
    receipt = entry.get("receipt")
    if not isinstance(receipt, str) or "/" in receipt or not _epoch_receipt_name(receipt):
        raise RehearsalRefusal("epoch heads pin names a bad receipts receipt")
    epoch_path = _receipt_member_path(root, receipt)
    if epoch_path.is_symlink() or not epoch_path.is_file():
        raise RehearsalRefusal("pinned receipts epoch is not a regular file")
    try:
        payload = json.loads(epoch_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RehearsalRefusal("pinned receipts epoch is unreadable") from exc
    if not isinstance(payload, dict):
        raise RehearsalRefusal("pinned receipts epoch is not an object")
    return _epoch_member_digests(payload)


def _epoch_member_digests(payload: Mapping[str, object]) -> dict[str, str]:
    members = payload.get("members")
    if not isinstance(members, list):
        raise RehearsalRefusal("pinned receipts epoch has no members list")
    out: dict[str, str] = {}
    for entry in members:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        digest = entry.get("sha256")
        if isinstance(name, str) and isinstance(digest, str) and _sha256_hex(digest):
            out[name] = digest
    return out


def _receipt_member_path(root: Path, name: str) -> Path:
    if (
        not name
        or name.startswith(("/", "\\"))
        or "\\" in name
        or "\x00" in name
        or "\n" in name
        or "\r" in name
    ):
        raise RehearsalRefusal(f"refusing receipt member {name!r}")
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise RehearsalRefusal(f"refusing receipt member {name!r}")
    link = root / "receipts"
    base = link.resolve()
    if link.is_symlink() or not base.is_dir():
        raise RehearsalRefusal("refusing a receipts directory that is not a real directory")
    candidate = base / rel
    if candidate.is_symlink():
        raise RehearsalRefusal(f"refusing symlink receipt member {name!r}")
    parent = candidate.parent.resolve()
    if parent != base and base not in parent.parents:
        raise RehearsalRefusal(f"refusing receipt member {name!r}")
    return parent / candidate.name


def _replace_receipt(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise RehearsalRefusal("refusing to write a symlink receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.restore")
    if tmp.exists() or tmp.is_symlink():
        raise RehearsalRefusal("refusing to replace over a restore temp")
    tmp.write_bytes(data)
    if path.is_symlink():
        tmp.unlink()
        raise RehearsalRefusal("refusing to write a symlink receipt")
    os.replace(tmp, path)


def _epoch_receipt_name(name: str) -> bool:
    return Path(name).name.startswith("corpus_epoch_") and name.endswith(".json")


def _sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _git_rev(rev: bytes) -> bool:
    text = rev.decode("ascii", errors="ignore")
    return text.encode("ascii") == rev and len(text) >= 7 and text.isalnum()


def write_quorum_registry(pubkey_hex: str, path: Path) -> None:
    """Write a threshold-1 gate_quorum.v1 registry containing one public key."""
    pub = pubkey_hex.strip()
    registry = {
        "schema": "gate_quorum.v1",
        "threshold": 1,
        "keys": [{"key_id": key_id(pub), "pubkey": pub}],
    }
    errors = quorum_registry_errors(registry)
    if errors:
        raise ValueError(f"quorum registry rejected: {errors}")
    path.write_bytes(registry_file_bytes(registry))


def main(argv: list[str] | None = None) -> int:
    """Print a public key, write a registry, or run the throwaway rehearsal."""
    parser = argparse.ArgumentParser(description="Throwaway owner-ceremony rehearsal")
    parser.add_argument("--public-hex", metavar="ENV")
    parser.add_argument("--quorum-registry", metavar="PATH")
    parser.add_argument("--restore-stamped-receipts", action="store_true")
    args = parser.parse_args(argv)
    if args.restore_stamped_receipts:
        if args.public_hex or args.quorum_registry:
            parser.error("--restore-stamped-receipts does not combine with other modes")
        return _restore_cli()
    if args.public_hex or args.quorum_registry:
        return _helpers(args.public_hex, args.quorum_registry)
    return rehearse()


def _restore_cli() -> int:
    root = _git_root(Path.cwd())
    restored, missing = restore_stamped_receipts(root)
    for name in restored:
        print(f"restored {name}")
    if missing:
        print("unrestored " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


def _helpers(public_env: str | None, registry_path: str | None) -> int:
    if public_env:
        print(public_hex(os.environ[public_env]))
    if registry_path:
        write_quorum_registry(os.environ["GATE_SIGNING_PUBKEY_NEW"], Path(registry_path))
    return 0


def rehearse() -> int:
    """Copy the checkout, run OWNER_STEPS there, then score that copy."""
    source = _git_root(Path.cwd())
    dest = Path(tempfile.mkdtemp(prefix="owner-rehearsal-", dir="/tmp"))
    keys = Path(tempfile.mkdtemp(prefix="owner-rehearsal-keys-", dir="/tmp"))
    assert_throwaway(source, dest)
    assert_throwaway(source, keys)
    before = _snapshot(source)
    code = 1
    try:
        code = _run_rehearsal(source, dest, keys)
    finally:
        shutil.rmtree(keys, ignore_errors=True)
        _assert_unchanged(source, before)
    return code


def _run_rehearsal(source: Path, dest: Path, keys: Path) -> int:
    print(f"owner-rehearsal: copy {dest}", flush=True)
    _copy_and_sync(source, dest)
    _isolate_git(dest)
    (dest / _MARKER).write_text("throwaway\n", encoding="utf-8")
    env = _child_env()
    _assert_imports_from_copy(source, dest, env)
    material = _mint(keys)
    require_copy(source, dest)
    _install_outgoing(dest, material)
    _open_spine(source, dest, material, env)
    print("owner-rehearsal: ceremony", flush=True)
    code = _run(dest, ["bash", "OWNER_STEPS.md"], _ceremony_env(env, material), source)
    if code != 0:
        print(f"owner-rehearsal: ceremony exited {code}", flush=True)
        return code
    retarget_data_manifest(source, dest)
    print("owner-rehearsal: prod-readiness", flush=True)
    score_env = _ceremony_env(env, material)
    score_env["PROD_READINESS_OUT"] = "/tmp/owner-rehearsal-scorecard.json"
    return _run(dest, ["make", "prod-readiness"], score_env, source)


def _copy_and_sync(source: Path, dest: Path) -> None:
    _copy_tree(source, dest)
    _link_venv(source, dest)


def _copy_tree(source: Path, dest: Path) -> None:
    """Copy the checkout, leaving .venv behind. tar is used because rsync is absent."""
    reader = subprocess.Popen(
        ["tar", "-C", str(source), "--exclude", ".venv", "-cf", "-", "."],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(["tar", "-C", str(dest), "-xf", "-"], stdin=reader.stdout)
    if reader.stdout is not None:
        reader.stdout.close()
    writer_rc = writer.wait()
    reader_rc = reader.wait()
    if reader_rc != 0 or writer_rc != 0:
        raise RehearsalRefusal(f"copy failed reader={reader_rc} writer={writer_rc}")


def _link_venv(source: Path, dest: Path) -> None:
    """Hardlink the source venv, then retarget the editable install at dest.

    Rewritten files are unlinked first so the source venv's inodes stay put.
    Later commands run with UV_NO_SYNC=1 and must not rewrite the link farm.
    """
    src_venv = source / ".venv"
    dst_venv = dest / ".venv"
    subprocess.run(["cp", "-al", str(src_venv), str(dst_venv)], check=True)
    for path in dst_venv.glob("lib/python*/site-packages/_editable_impl_*.pth"):
        _retarget_file(path, str(source / "src"), str(dest / "src"))
    for script in (dst_venv / "bin").iterdir():
        if script.is_symlink() or not script.is_file():
            continue
        _retarget_file(script, str(src_venv), str(dst_venv))


def _retarget_file(path: Path, old: str, new: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    if old not in text:
        return
    mode = path.stat().st_mode
    path.unlink()
    path.write_text(text.replace(old, new), encoding="utf-8")
    path.chmod(mode)


def _isolate_git(dest: Path) -> None:
    subprocess.run(
        ["git", "-C", str(dest), "remote", "remove", "origin"],
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["git", "-C", str(dest), "config", "commit.gpgsign", "false"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "config", "tag.gpgsign", "false"],
        check=True,
    )
    listed = subprocess.run(
        ["git", "-C", str(dest), "remote"],
        capture_output=True,
        text=True,
        check=True,
    )
    if listed.stdout.strip():
        raise RehearsalRefusal("throwaway still has a git remote")
    _assert_no_credential(dest)


def _assert_no_credential(dest: Path) -> None:
    text = (dest / ".git" / "config").read_text(encoding="utf-8", errors="replace")
    if any(needle in text for needle in ("ghs_", "ghp_", "github_pat_", "x-access-token")):
        raise RehearsalRefusal("throwaway git config still has a credential")


def _assert_imports_from_copy(source: Path, dest: Path, env: dict[str, str]) -> None:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            "import quant_fund; print(quant_fund.__file__)",
        ],
        cwd=dest,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    location = Path(proc.stdout.strip()).resolve()
    if not location.is_relative_to(dest.resolve()):
        raise RehearsalRefusal("interpreter is not the throwaway copy")
    if location.is_relative_to(source.resolve()):
        raise RehearsalRefusal("interpreter resolved inside the real checkout")


def _install_outgoing(dest: Path, material: dict[str, str]) -> None:
    pub = material["pub_a"]
    (dest / "quality" / "gate_signing.pub").write_text(pub + "\n", encoding="utf-8")
    init_quorum(dest, [pub], threshold=1)
    (dest / "quality" / "witness_signing.pub").write_text(material["witness_pub"], encoding="utf-8")


def _open_spine(source: Path, dest: Path, material: dict[str, str], env: dict[str, str]) -> None:
    """Checkpoint and witness the outgoing key so rotation has a genesis."""
    require_copy(source, dest)
    clear_spine(dest)
    boot = _ceremony_env(env, material)
    boot["GATE_SIGNING_KEY"] = f"{material['seed_a']}:{material['pub_a']}"
    code = _run(dest, ["uv", "run", "--no-sync", "dipcatcher", "checkpoint"], boot, source)
    if code != 0:
        raise RehearsalRefusal(f"bootstrap checkpoint exited {code}")
    code = _run(dest, ["uv", "run", "--no-sync", "dipcatcher", "witness-checkpoint"], boot, source)
    if code != 0:
        raise RehearsalRefusal(f"bootstrap witness exited {code}")


def clear_spine(dest: Path) -> None:
    """Drop checkpoint and witness files inside dest so an ephemeral key can sign."""
    quality = dest / "quality"
    paths = list((quality / "witness").glob("*.json"))
    paths.extend((quality / "checkpoints").glob("*.json"))
    checkpoint = quality / "checkpoint.json"
    if checkpoint.is_file():
        paths.append(checkpoint)
    for path in paths:
        unlink_under(quality, path)


def _mint(keys: Path) -> dict[str, str]:
    os.chmod(keys, stat.S_IRWXU)
    seed_a, pub_a = generate_keypair()
    seed_b, pub_b = generate_keypair()
    witness_pem, witness_pub = _witness_key()
    pem_path = keys / "witness.pem"
    pem_path.write_bytes(witness_pem)
    os.chmod(pem_path, stat.S_IRUSR | stat.S_IWUSR)
    card = keys / "modelcard.json"
    weights = keys / "weights.safetensors"
    card.write_text(_card_text(), encoding="utf-8")
    weights.write_bytes(b"fx1-rehearsal-weights")
    return {
        "seed_a": seed_a,
        "pub_a": pub_a,
        "seed_b": seed_b,
        "pub_b": pub_b,
        "witness_pem": witness_pem.decode(),
        "witness_pub": witness_pub.decode(),
        "witness_path": str(pem_path),
        "card": str(card),
        "weights": str(weights),
    }


def _card_text() -> str:
    return json.dumps(_CARD, indent=2, sort_keys=True) + "\n"


def _witness_key() -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return pem, pub


def _ceremony_env(base: dict[str, str], material: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env["OWNER_REHEARSAL"] = "1"
    env["UV_NO_SYNC"] = "1"
    env["GATE_SIGNING_KEY"] = material["seed_a"]
    env["GATE_SIGNING_KEY_NEW"] = material["seed_b"]
    env["WITNESS_SIGNING_KEY"] = material["witness_path"]
    env["FX1_PUBLISHED_CARD"] = material["card"]
    env["FX1_PUBLISHED_WEIGHTS"] = material["weights"]
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_EDITOR"] = "true"
    return env


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.pop("VIRTUAL_ENV", None)
    env["UV_NO_SYNC"] = "1"
    return env


def _run(dest: Path, argv: list[str], env: dict[str, str], source: Path) -> int:
    require_copy(source, dest)
    log_path = Path("/tmp/owner-rehearsal.log")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(argv)}\n")
        proc = subprocess.run(
            argv,
            cwd=dest,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        _show_log_tail(log_path, env)
    return proc.returncode


def _show_log_tail(log_path: Path, env: dict[str, str]) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    redacted = _redact(text, env)
    tail = "\n".join(redacted.splitlines()[-40:])
    print(tail, file=sys.stderr, flush=True)


def _redact(text: str, env: dict[str, str]) -> str:
    import re

    for name in ("GATE_SIGNING_KEY", "GATE_SIGNING_KEY_NEW"):
        secret = env.get(name, "")
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[redacted]",
        text,
        flags=re.DOTALL,
    )
    return text


def _git_root(start: Path) -> Path:
    proc = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip())


def _snapshot(source: Path) -> str:
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    digests = []
    for rel in _SENTINELS:
        path = source / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        digests.append(f"{rel} {digest}")
    return status.stdout + "\n" + "\n".join(digests)


def _assert_unchanged(source: Path, before: str) -> None:
    if _snapshot(source) != before:
        raise RehearsalRefusal("owner-rehearsal changed the real checkout")


if __name__ == "__main__":
    raise SystemExit(main())
