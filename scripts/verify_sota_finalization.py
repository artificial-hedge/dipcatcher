#!/usr/bin/env python3
"""Independent verifier for SOTA finalization manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class VerificationError(ValueError):
    """Manifest or evidence integrity failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"JSON root is not an object: {path}")
    return value


def _required(obj: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    value = obj.get(key)
    if not isinstance(value, kind):
        raise VerificationError(f"missing or invalid field {key!r}")
    return value


def _windows_absolute(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def _recorded_candidates(value: str | os.PathLike[str], manifest_dir: Path, repo: str | None = None) -> list[Path]:
    """Return lexical candidates without dereferencing links."""
    raw = os.fspath(value)
    candidates: list[Path] = []
    native = Path(raw)
    if native.is_absolute():
        candidates.append(native)
    else:
        candidates.extend((manifest_dir / native, Path.cwd() / native))
    if _windows_absolute(raw):
        slash = raw.replace("\\", "/")
        drive, tail = slash[0].lower(), slash[2:].lstrip("/")
        candidates.append(Path("/") / drive / tail)
        if repo and _windows_absolute(repo):
            repo_slash = repo.replace("\\", "/").rstrip("/")
            if slash.lower().startswith(repo_slash.lower() + "/"):
                candidates.append(Path.cwd() / slash[len(repo_slash):].lstrip("/"))
    return [candidate.expanduser().absolute() for candidate in candidates]


def _resolve_recorded(value: str | os.PathLike[str], manifest_dir: Path, repo: str | None = None) -> Path:
    """Resolve native paths and Windows paths emitted by PowerShell."""
    candidates = _recorded_candidates(value, manifest_dir, repo)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(strict=False)
    return candidates[0].resolve(strict=False)


def _resolve_recorded_lexical(value: str | os.PathLike[str], manifest_dir: Path, repo: str | None = None) -> Path:
    """Resolve a recorded path while preserving link/reparse components."""
    candidates = _recorded_candidates(value, manifest_dir, repo)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _provenance_key(value: str) -> str:
    """Normalize a recorded path key without resolving filesystem links."""
    key = value.replace("\\", "/")
    if _windows_absolute(key):
        return key.lower()
    return key


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_reparse_any(path: Path) -> bool:
    """Return whether any existing component of a path is a link/reparse point."""
    absolute = path.absolute()
    current = Path(absolute.anchor) if absolute.anchor else Path('.')
    for part in absolute.parts:
        if part == absolute.anchor:
            continue
        current /= part
        try:
            if current.is_symlink():
                return True
            if os.name == "nt":
                attributes = os.stat(current, follow_symlinks=False).st_file_attributes
                if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                    return True
        except OSError:
            return True
    return False


def _has_reparse_component(path: Path, root: Path) -> bool:
    """Return whether an existing path component is a link/reparse point."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    components = (current, *relative.parts)
    for part in components:
        current = Path(part) if current == root and part == current else current / part
        try:
            if current.is_symlink():
                return True
            if os.name == "nt":
                attributes = os.stat(current, follow_symlinks=False).st_file_attributes
                if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                    return True
        except OSError:
            return True
    return False


def _resolve_artifact(
    value: str | os.PathLike[str],
    manifest_dir: Path,
    repo: str | None,
    *,
    label: str,
) -> Path:
    """Resolve an artifact and reject escapes or link/reparse aliases."""
    lexical = _resolve_recorded_lexical(value, manifest_dir, repo)
    run_root = manifest_dir.absolute()
    if not _under(lexical, run_root):
        raise VerificationError(f"{label} escapes manifest run directory: {value}")
    if _has_reparse_component(lexical, run_root):
        raise VerificationError(f"{label} uses a symlink or reparse point: {value}")
    resolved = lexical.resolve(strict=False)
    if not lexical.exists():
        return lexical
    if not _under(resolved, run_root):
        raise VerificationError(f"{label} resolves outside manifest run directory: {value}")
    return resolved


def _verify_shards(group: dict[str, Any], input_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shards = _required(group, "shards", list)
    if not shards:
        raise VerificationError("shards must not be empty")
    manifest_dir = Path(manifest["_manifest_dir"])
    seen_paths: set[Path] = set()
    seen_relative: set[str] = set()
    seen_hashes: set[str] = set()
    verified: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise VerificationError(f"shard {index} is not an object")
        path_text = _required(shard, "path", str)
        relative = _required(shard, "relative", str).replace("\\", "/")
        expected_hash = _required(shard, "sha256", str).lower()
        expected_bytes = _required(shard, "bytes", int)
        if not _SHA256.fullmatch(expected_hash):
            raise VerificationError(f"invalid shard hash: {path_text}")
        relative_parts = Path(relative).parts
        if Path(relative).is_absolute() or _windows_absolute(relative) or ".." in relative_parts:
            raise VerificationError(f"shard relative path escapes input: {relative}")
        actual_lexical = _resolve_recorded_lexical(path_text, manifest_dir, manifest.get("repo"))
        repo_text = manifest.get("repo")
        if not isinstance(repo_text, str):
            raise VerificationError("manifest repo is required for shard relative paths")
        repo_root = _resolve_recorded(repo_text, manifest_dir)
        expected_lexical = repo_root / relative
        if not _under(actual_lexical, input_root) or not _under(expected_lexical, input_root):
            raise VerificationError(f"shard path/relative mismatch or escape: {path_text}")
        if not actual_lexical.exists():
            raise VerificationError(f"missing shard: {path_text}")
        if _has_reparse_component(actual_lexical, input_root) or _has_reparse_component(expected_lexical, input_root):
            raise VerificationError(f"shard uses a symlink or reparse point: {path_text}")
        actual = actual_lexical.resolve(strict=False)
        expected = expected_lexical.resolve(strict=False)
        if actual != expected or not _under(actual, input_root) or not _under(expected, input_root):
            raise VerificationError(f"shard path/relative mismatch or escape: {path_text}")
        if actual in seen_paths or relative.lower() in seen_relative or expected_hash in seen_hashes:
            raise VerificationError(f"duplicate shard metadata: {path_text}")
        seen_paths.add(actual)
        seen_relative.add(relative.lower())
        seen_hashes.add(expected_hash)
        if not actual.is_file():
            raise VerificationError(f"missing shard: {path_text}")
        if actual.stat().st_size != expected_bytes:
            raise VerificationError(f"shard byte-size mismatch: {path_text}")
        actual_hash = _sha256(actual)
        if actual_hash != expected_hash:
            raise VerificationError(f"shard hash mismatch: {path_text}")
        verified.append({"path": actual, "relative": relative, "sha256": actual_hash})
    return verified


def _verify_execution(group: dict[str, Any], manifest: dict[str, Any]) -> None:
    directory = Path(manifest["_manifest_dir"])
    command_record = _resolve_artifact(
        _required(group, "command_record", str), directory, manifest.get("repo"), label="command record"
    )
    if not command_record.is_file():
        raise VerificationError(f"missing command record for {group.get('group')}: {command_record}")
    recorded = _load_json(command_record)
    command = _required(group, "command", list)
    if not command or not all(isinstance(item, str) and item for item in command):
        raise VerificationError(f"invalid command for {group.get('group')}")
    command_text = _required(group, "command_text", str)
    if command_text != " ".join(command):
        raise VerificationError(f"command_text mismatch for {group.get('group')}")
    record_fields = [
        "group", "command", "command_text", "cwd", "shards", "stdout", "stderr", "transcript",
        "receipt", "receipt_sha256", "losses", "losses_sha256", "exit_code", "status",
    ]
    if manifest.get("schema") == "sota_finalization.v2":
        record_fields.extend(("stdout_sha256", "stderr_sha256", "transcript_sha256"))
    for field in record_fields:
        if recorded.get(field) != group.get(field):
            raise VerificationError(f"command record mismatch for {group.get('group')}: {field}")
    repo = _resolve_recorded(_required(manifest, "repo", str), directory)
    cwd = _resolve_recorded(_required(group, "cwd", str), directory, manifest.get("repo"))
    if cwd != repo:
        raise VerificationError(f"command cwd is not repository root for {group.get('group')}")
    for key in ("stdout", "stderr", "transcript"):
        output = _resolve_artifact(_required(group, key, str), directory, manifest.get("repo"), label=key)
        if not output.is_file():
            raise VerificationError(f"missing {key} artifact for {group.get('group')}: {output}")
        if manifest.get("schema") == "sota_finalization.v2":
            expected_hash = _required(group, f"{key}_sha256", str).lower()
            if not _SHA256.fullmatch(expected_hash) or _sha256(output) != expected_hash:
                raise VerificationError(f"{key} hash mismatch: {output}")
    transcript_path = _resolve_artifact(group["transcript"], directory, manifest.get("repo"), label="transcript")
    transcript = transcript_path.read_text(encoding="utf-8-sig")
    lines = transcript.splitlines()
    if len(lines) < 2 or lines[0] != f"COMMAND: {command_text}":
        raise VerificationError(f"transcript command does not bind recorded argv for {group.get('group')}")
    match = re.fullmatch(r"EXIT_CODE: (-?\d+)", lines[1])
    if match is None or int(match.group(1)) != 0 or group.get("exit_code") != 0:
        raise VerificationError(f"transcript does not bind command and successful exit for {group.get('group')}")


def _inspect_losses(path: Path) -> tuple[int, int, tuple[int, ...]]:
    if np is None:
        raise VerificationError("numpy is required for NPZ validation")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required_names = {"crps_matrix", "pinball_cube", "asset_ids", "model_names", "meta_json"}
            missing = required_names.difference(archive.files)
            if missing:
                raise VerificationError(f"loss archive missing arrays: {sorted(missing)}")
            crps = archive["crps_matrix"]
            pinball = archive["pinball_cube"]
            if crps.ndim != 2 or pinball.ndim != 3 or pinball.shape[:2] != crps.shape or pinball.shape[2] != 3:
                raise VerificationError("loss arrays must be crps(n,m) and pinball(n,m,3)")
            if len(archive["asset_ids"]) != crps.shape[0] or len(archive["model_names"]) != crps.shape[1]:
                raise VerificationError("loss metadata dimensions mismatch")
            meta = archive["meta_json"]
            if meta.ndim != 0:
                raise VerificationError("meta_json must be scalar")
            json.loads(str(meta.item()))
            finite = np.isfinite(crps).all(axis=1) & np.isfinite(pinball).all(axis=(1, 2))
            return int(crps.shape[0]), int(finite.sum()), tuple(int(v) for v in crps.shape)
    except VerificationError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot inspect loss archive {path}: {exc}") from exc


def _verify_receipt(group: dict[str, Any], shards: list[dict[str, Any]], manifest: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    directory = Path(manifest["_manifest_dir"])
    receipt = _resolve_artifact(_required(group, "receipt", str), directory, manifest.get("repo"), label="receipt")
    receipt_hash = _required(group, "receipt_sha256", str).lower()
    if not receipt.is_file() or not _SHA256.fullmatch(receipt_hash) or _sha256(receipt) != receipt_hash:
        raise VerificationError(f"receipt missing or hash mismatch: {receipt}")
    data = _load_json(receipt)
    if data.get("schema") != "sota_eval.v4":
        raise VerificationError(f"unsupported receipt schema: {data.get('schema')!r}")
    if data.get("research_only") is not True or data.get("live_pnl_claim") is not False:
        raise VerificationError("receipt is not explicitly research-only")
    source = data.get("source_parts_sha256")
    if not isinstance(source, dict) or len(source) != len(shards):
        raise VerificationError("receipt source-part hash map is invalid")
    expected_source = {_provenance_key(str(s["path"])): s["sha256"] for s in shards}
    normalized_source = {_provenance_key(str(key)): str(value).lower() for key, value in source.items()}
    if normalized_source != expected_source:
        raise VerificationError("receipt source-part paths or hashes do not match manifest shards")
    losses = _resolve_artifact(_required(group, "losses", str), directory, manifest.get("repo"), label="losses")
    losses_hash = _required(group, "losses_sha256", str).lower()
    if not losses.is_file() or not _SHA256.fullmatch(losses_hash) or _sha256(losses) != losses_hash:
        raise VerificationError(f"loss archive missing or hash mismatch: {losses}")
    if data.get("losses_file") != losses.name or data.get("losses_sha256") != losses_hash:
        raise VerificationError("receipt loss provenance does not match manifest")
    n_rows, n_complete, shape = _inspect_losses(losses)
    if data.get("n_rows") != n_rows or data.get("n_complete") != n_complete or data.get("n_dropped_incomplete") != n_rows - n_complete:
        raise VerificationError("receipt completeness counts do not match loss arrays")
    if n_complete < n_rows:
        warnings.append(f"{group['group']}: incomplete rows {n_complete}/{n_rows}")
    if data.get("scoring_contract") != "verified.v1":
        warnings.append(f"{group['group']}: scoring contract is unverified")
    return {"group": group["group"], "status": "succeeded", "n_rows": n_rows, "n_complete": n_complete, "loss_shape": shape, "scoring_contract": data.get("scoring_contract")}


def verify_manifest(manifest_path: str | os.PathLike[str], *, proof_grade: bool = False) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().absolute()
    errors: list[str] = []
    warnings: list[str] = []
    groups_report: list[dict[str, Any]] = []
    try:
        if not path.is_file() or _has_reparse_any(path):
            raise VerificationError(f"manifest uses a symlink or reparse point or is missing: {path}")
        manifest = _load_json(path)
        manifest["_manifest_dir"] = str(path.parent)
        if manifest.get("schema") not in {"sota_finalization.v1", "sota_finalization.v2"}:
            raise VerificationError("unsupported manifest schema")
        if manifest.get("proof_status") != "UNPROVEN":
            raise VerificationError("manifest proof_status must remain UNPROVEN")
        run_id = _required(manifest, "run_id", str)
        if not _SAFE_RUN_ID.fullmatch(run_id) or path.parent.name != run_id:
            raise VerificationError("unsafe or directory-mismatched run_id")
        repo_text = _required(manifest, "repo", str)
        repo_lexical = _resolve_recorded_lexical(repo_text, path.parent)
        if not repo_lexical.is_dir():
            raise VerificationError(f"missing repository directory: {repo_lexical}")
        if _has_reparse_any(repo_lexical):
            raise VerificationError(f"repository uses a symlink or reparse point: {repo_lexical}")
        input_text = _required(manifest, "input_dir", str)
        input_lexical = _resolve_recorded_lexical(input_text, path.parent, repo_text)
        if not input_lexical.is_dir():
            raise VerificationError(f"missing input directory: {input_lexical}")
        if not _under(input_lexical, repo_lexical):
            raise VerificationError(f"input directory escapes repository: {input_text}")
        if _has_reparse_any(input_lexical):
            raise VerificationError(f"input directory uses a symlink or reparse point: {input_lexical}")
        expected_run = input_lexical / "runs" / run_id
        if path.parent != expected_run:
            raise VerificationError("manifest is not under repository input_dir/runs/run_id")
        if _has_reparse_any(path.parent):
            raise VerificationError(f"manifest run directory uses a symlink or reparse point: {path.parent}")
        input_root = input_lexical.resolve(strict=False)
        groups = _required(manifest, "groups", list)
        if not groups:
            raise VerificationError("manifest groups must not be empty")
        names: set[str] = set()
        all_executed = True
        for group in groups:
            if not isinstance(group, dict):
                raise VerificationError("group is not an object")
            name = _required(group, "group", str)
            if name in names:
                raise VerificationError(f"duplicate group: {name}")
            names.add(name)
            status = group.get("status")
            if status == "succeeded":
                if group.get("exit_code") != 0:
                    raise VerificationError(f"succeeded group has nonzero exit: {name}")
                shards = _verify_shards(group, input_root, manifest)
                groups_report.append(_verify_receipt(group, shards, manifest, warnings))
                _verify_execution(group, manifest)
            elif status == "not_run_verify_only":
                all_executed = False
                _verify_shards(group, input_root, manifest)
                warnings.append(f"{name}: verify-only preflight has no receipt")
                groups_report.append({"group": name, "status": status})
            else:
                raise VerificationError(f"unsupported group status for {name}: {status!r}")
        if manifest.get("status") == "complete" and not all_executed:
            raise VerificationError("complete manifest contains a non-executed group")
        if manifest.get("status") not in {"complete", "verify_only"}:
            raise VerificationError(f"unsupported manifest status: {manifest.get('status')!r}")
    except (VerificationError, KeyError, TypeError, OSError) as exc:
        errors.append(str(exc))
    if proof_grade:
        errors.extend(f"proof-grade rejection: {warning}" for warning in warnings)
        if not errors:
            errors.append("proof-grade SOTA status is unavailable: proof_status is UNPROVEN")
    return {
        "valid": not errors,
        "classification": "invalid" if errors else "diagnostic_only",
        "proof_eligible": False,
        "errors": errors,
        "warnings": warnings,
        "groups": groups_report,
        "manifest": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--proof-grade", action="store_true")
    args = parser.parse_args(argv)
    result = verify_manifest(args.manifest, proof_grade=args.proof_grade)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and not args.proof_grade else 1


if __name__ == "__main__":
    raise SystemExit(main())
