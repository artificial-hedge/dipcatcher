"""Replay proofs: re-execute a lane's declared argv and verify artifact bytes.

``verify-receipt`` proves a receipt's seal and re-derives claims *inside* the
payload; it never re-runs the lane. The honesty contract asks for more: every
research claim should be reproducible from a receipt hash. A **replay
manifest** closes that loop — a receipt may carry an optional ``replay``
block declaring the exact argv that produces its artifacts plus the input
tapes the lane read::

    "replay": {
        "argv": ["dipcatcher", "serial-watch", "tests/fixtures/replay/serial_pits.json",
                 "--data-label", "SYNTHETIC", "--out-dir", "data/metadata/replay"],
        "artifacts": [{"path": "data/metadata/replay/serial_watch_<d16>.json",
                       "sha256": "<64-hex file digest>"}],
        "cwd": ".",                        # optional, relative to repo root
        "input_tapes": [{"path": "data/raw/x/bars.parquet", "sha256": "<64-hex>"},
                        {"manifest": "data/manifests/yahoo_eod.json"}]
    }

``input_tapes`` (optional) binds the tape bytes the replayed lane read.
Each entry is a literal ``{path, sha256}`` file pin, or a ``{manifest}``
reference to a committed ``tape_manifest.v1`` — expanded to its declared
``tape_files`` so the input digests come from committed evidence, not the
replay block itself. Inputs are verified **before** argv runs: a missing
or drifted tape yields ``inputs_ok=false`` and a ``fail`` verdict without
executing — a tampered tape must never reach a lane's parser, and the
proof then attributes the failure (``input_tape_missing`` /
``input_tape_drift``) instead of a bare artifact mismatch.

``replay_manifest`` reads the block (``None`` when absent — an undeclared
lane is honest, it simply has nothing to re-execute). ``run_replay``
executes argv as a timeout-bounded subprocess, re-hashes each declared
artifact, and emits a ``replay_proof.v1`` body. **Fail closed everywhere:**
a non-zero exit, a timeout, a spawn error, a missing artifact, or any
digest mismatch yields ``verdict="fail"`` — the proof attests what was
observed, never what was hoped.

The body carries ``receipt_sha256`` as a canonical self-seal over the
proof body itself — the same convention every other receipt uses, and the
only semantics ``receipt.v2`` verification accepts for that key inside a
wrapped payload. The proof's binding to the *source* receipt lives in
``receipt`` (its path) plus ``source_receipt_sha256`` (its file digest)
and ``source_receipt_seal`` (the seal the lane stamped, when present).

Path discipline: the optional ``cwd`` and every relative artifact path are
resolved under ``root`` and must stay inside it — a manifest that escapes
the checkout fails rather than following a receipt outside its evidence
boundary. ``argv[0] == "dipcatcher"`` (or ``quant``) is resolved to the
current interpreter's ``quant_fund.cli.main`` entry point so the proof
exercises the code under audit, not whatever happens to be on ``PATH``
(pyproject maps both names to ``quant_fund.cli.main:app``). The declared
argv is recorded verbatim next to the resolved form for audit.

A receipt cannot declare itself among ``artifacts`` — a file cannot embed
its own digest — so lanes emitting replay manifests declare the sibling
artifacts argv produces (e.g. the digest-named receipt files), which are
necessarily different files than the manifest-bearing resealed copy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPLAY_PROOF_SCHEMA = "replay_proof.v1"
REPLAY_MANIFEST_KEY = "replay"
REPLAY_PROOF_VERDICTS = ("pass", "fail")
DEFAULT_TIMEOUT_S = 120.0
_MAX_STDERR_TAIL = 2048

#: Console-script names pyproject maps to ``quant_fund.cli.main:app``.
_CLI_ENTRYPOINTS = frozenset({"dipcatcher", "quant"})
_SHA256_LEN = 64

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "REPLAY_MANIFEST_KEY",
    "REPLAY_PROOF_SCHEMA",
    "REPLAY_PROOF_VERDICTS",
    "replay_manifest",
    "replay_manifest_errors",
    "replay_proof_contract_errors",
    "run_replay",
]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LEN
        and all(character in "0123456789abcdef" for character in value)
    )


def replay_manifest_errors(manifest: object) -> list[str]:
    """Structural errors in a declared ``replay`` block; ``[]`` when well-formed."""
    if not isinstance(manifest, Mapping):
        return ["manifest_not_object"]
    errors: list[str] = []
    argv = manifest.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(arg, str) and arg for arg in argv)
    ):
        errors.append("argv_not_nonempty_str_list")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts_not_nonempty_list")
    else:
        for index, entry in enumerate(artifacts):
            if not isinstance(entry, Mapping):
                errors.append(f"artifacts[{index}]_not_object")
                continue
            path = entry.get("path")
            if not isinstance(path, str) or not path.strip():
                errors.append(f"artifacts[{index}].path")
            if not _is_sha256(entry.get("sha256")):
                errors.append(f"artifacts[{index}].sha256")
    cwd = manifest.get("cwd")
    if cwd is not None and not (isinstance(cwd, str) and cwd.strip()):
        errors.append("cwd_not_nonempty_str")
    tapes = manifest.get("input_tapes")
    if tapes is not None:
        if not isinstance(tapes, list):
            errors.append("input_tapes_not_list")
        else:
            for index, entry in enumerate(tapes):
                if not isinstance(entry, Mapping):
                    errors.append(f"input_tapes[{index}]_not_object")
                    continue
                has_pin = isinstance(entry.get("path"), str) and bool(entry["path"].strip())
                has_manifest = isinstance(entry.get("manifest"), str) and bool(
                    entry["manifest"].strip()
                )
                if has_pin == has_manifest:
                    errors.append(f"input_tapes[{index}]_needs_exactly_one_of_path_or_manifest")
                    continue
                if has_pin and not _is_sha256(entry.get("sha256")):
                    errors.append(f"input_tapes[{index}].sha256")
    return errors


def replay_manifest(receipt_path: Path | str) -> dict[str, Any] | None:
    """Return a receipt's declared ``replay`` manifest, or ``None`` when absent.

    A missing block means the lane is not replay-declared — honest, not a
    failure. A present-but-malformed block is a broken declaration and
    raises ``ValueError`` (fail closed: a corrupt manifest must not produce
    a half-proof). The returned dict is a normalized copy:
    ``{argv: [...], artifacts: [{path, sha256}, ...], cwd?: str}``.
    """
    path = Path(receipt_path)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read receipt {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    # A receipt.v2 envelope wraps the lane body under ``payload`` — look at
    # both levels so wrapped lanes stay replay-declarable.
    manifest = body.get(REPLAY_MANIFEST_KEY)
    if manifest is None and isinstance(body.get("payload"), Mapping):
        manifest = body["payload"].get(REPLAY_MANIFEST_KEY)
    if manifest is None:
        return None
    errors = replay_manifest_errors(manifest)
    if errors:
        raise ValueError(f"replay manifest in {path} is malformed: {errors}")
    declared: dict[str, Any] = {
        "argv": [str(arg) for arg in manifest["argv"]],
        "artifacts": [
            {"path": str(entry["path"]), "sha256": str(entry["sha256"])}
            for entry in manifest["artifacts"]
        ],
    }
    if manifest.get("cwd") is not None:
        declared["cwd"] = str(manifest["cwd"])
    if manifest.get("input_tapes") is not None:
        declared["input_tapes"] = [dict(entry) for entry in manifest["input_tapes"]]
    return declared


def _source_data_label(receipt_path: Path) -> str:
    """The label the replay covers — the source receipt's own ``data_label``."""
    try:
        body = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "UNKNOWN"
    if not isinstance(body, dict):
        return "UNKNOWN"
    for candidate in (body.get("data_label"), (body.get("payload") or {}).get("data_label")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return "UNKNOWN"


def _resolve_argv(argv: list[str]) -> list[str]:
    """Map the declared argv to the command actually executed.

    ``dipcatcher``/``quant`` resolve to the current interpreter's
    ``quant_fund.cli.main`` module — the entry point pyproject binds them
    to — so the replay runs this checkout's code rather than depending on
    ``PATH``. A leading ``*.py`` path resolves to the same interpreter so
    script lanes also run this checkout's code. Any other argv runs verbatim.
    """
    if argv and argv[0] in _CLI_ENTRYPOINTS:
        return [sys.executable, "-m", "quant_fund.cli.main", *argv[1:]]
    if argv and argv[0].endswith(".py"):
        return [sys.executable, *argv]
    return list(argv)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_input_tapes(
    declared: list[Mapping[str, Any]], root_path: Path
) -> tuple[list[dict[str, Any]], bool]:
    """Verify declared input tapes against committed bytes before execution.

    ``{path, sha256}`` entries pin one file; ``{manifest}`` entries expand to
    the tape_files of a seal-valid ``tape_manifest.v1`` under ``root_path`` —
    the manifest itself must verify or every derived row fails closed.
    Returns (rows, all_ok). Row ``note`` values attribute failure precisely:
    ``input_tape_missing``, ``input_tape_drift``, ``manifest_missing``,
    ``manifest_invalid``, ``path_escapes_root``.
    """
    from quant_fund.research.tape_registry import TAPE_MANIFEST_SCHEMA
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    rows: list[dict[str, Any]] = []
    for entry in declared:
        if "manifest" in entry:
            rel = str(entry["manifest"])
            manifest_path = (root_path / rel).resolve()
            if not manifest_path.is_relative_to(root_path):
                rows.append({"manifest": rel, "note": "path_escapes_root", "match": False})
                continue
            try:
                manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = None
            seal_ok = False
            tape_files: list[Mapping[str, Any]] = []
            if isinstance(manifest, dict) and manifest.get("schema") == TAPE_MANIFEST_SCHEMA:
                seal = manifest.get("receipt_sha256")
                body = {k: v for k, v in manifest.items() if k != "receipt_sha256"}
                seal_ok = _is_sha256(seal) and hash_bytes(canonical_json_bytes(body)) == seal
                tape_files = [
                    t for t in (manifest.get("tape_files") or []) if isinstance(t, Mapping)
                ]
            if not manifest_path.is_file():
                rows.append({"manifest": rel, "note": "manifest_missing", "match": False})
            elif not seal_ok:
                rows.append({"manifest": rel, "note": "manifest_invalid", "match": False})
            elif not tape_files:
                rows.append({"manifest": rel, "note": "manifest_no_tape_files", "match": False})
            for tape in tape_files:
                rows.append(
                    {
                        "path": str(tape.get("path")),
                        "expected_sha256": tape.get("sha256"),
                        "via_manifest": rel,
                        **_tape_file_row(tape, root_path),
                    }
                )
            continue
        rows.append(_tape_file_row(entry, root_path))
    return rows, all(row.get("match") is True for row in rows)


def _tape_file_row(entry: Mapping[str, Any], root_path: Path) -> dict[str, Any]:
    """Resolve one ``{path, sha256}`` tape pin against the live tree."""
    declared_path = str(entry.get("path") or "")
    expected = str(entry.get("sha256") or "")
    resolved = (root_path / declared_path).resolve()
    row: dict[str, Any] = {
        "path": declared_path,
        "expected_sha256": expected,
        "observed_sha256": None,
        "match": False,
    }
    if not _is_sha256(expected):
        row["note"] = "expected_sha256_invalid"
        return row
    if not resolved.is_relative_to(root_path):
        row["note"] = "path_escapes_root"
        return row
    if not resolved.is_file():
        row["note"] = "input_tape_missing"
        return row
    observed = _file_sha256(resolved)
    row["observed_sha256"] = observed
    row["match"] = observed == expected
    if not row["match"]:
        row["note"] = "input_tape_drift"
    return row


def _committed_artifact(
    artifacts: list[Mapping[str, Any]], root_path: Path, run_cwd: Path
) -> str | None:
    """Return the first declared artifact path that is git-tracked, or None.

    Replays may legitimately produce bytes identical to a committed file, but
    they must land on an untracked path; writing onto a tracked path would
    clobber the very evidence the manifest claims to reproduce.
    """
    rel_paths: list[str] = []
    resolved_entries: list[tuple[str, Path]] = []
    for entry in artifacts:
        declared = str(entry.get("path") or "")
        candidate = Path(declared)
        resolved = candidate if candidate.is_absolute() else (run_cwd / candidate)
        resolved = resolved.resolve()
        if not resolved.is_relative_to(root_path):
            continue  # escapes are flagged by the artifact pass itself
        rel = str(resolved.relative_to(root_path))
        rel_paths.append(rel)
        resolved_entries.append((declared, resolved))
    if not rel_paths:
        return None
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *rel_paths],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None  # not a git checkout — nothing to clobber
    tracked = set(proc.stdout.split())
    for declared, resolved in resolved_entries:
        if str(resolved.relative_to(root_path)) in tracked:
            return declared
    return None


def run_replay(
    receipt_path: Path | str,
    *,
    root: Path | str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute a receipt's declared replay and emit a ``replay_proof.v1`` body.

    Runs argv as a timeout-bounded subprocess under ``root`` (or the
    manifest's ``cwd`` inside it), re-hashes each declared artifact, and
    compares observed bytes against the manifest's pinned digests.
    ``all_match`` requires ``exit_code == 0`` AND every artifact matching;
    ``verdict`` is ``"pass"`` iff ``all_match`` — any deviation fails.

    Raises ``ValueError`` when the receipt is not replay-declared (no
    manifest) or the manifest is malformed — those outcomes attest nothing
    a proof body could honestly carry.
    """
    if not (isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool)):
        raise ValueError("timeout_s must be a real number")
    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
        raise ValueError("timeout_s must be finite and > 0")
    root_path = Path(root).resolve()
    path = Path(receipt_path)
    manifest = replay_manifest(path)
    if manifest is None:
        raise ValueError(f"receipt is not replay-declared: {path}")
    argv: list[str] = manifest["argv"]
    resolved_argv = _resolve_argv(argv)
    declared_cwd = manifest.get("cwd")
    run_cwd = (root_path / declared_cwd).resolve() if declared_cwd else root_path
    try:
        receipt_ref = str(path.resolve().relative_to(root_path))
    except (OSError, ValueError):
        receipt_ref = str(path)
    source_receipt_sha256 = _file_sha256(path)
    source_receipt_seal: str | None = None
    try:
        source_body = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(source_body, dict):
            seal = source_body.get("receipt_sha256")
            if _is_sha256(seal):
                source_receipt_seal = str(seal)
    except (OSError, UnicodeError, json.JSONDecodeError):
        source_receipt_seal = None

    env = dict(os.environ)
    # The replayed argv must run the verifier's own code, never whatever an
    # ambient installed package resolves to — prefer the root's src/, else
    # fall back to this process's own quant_fund source tree.
    src_dir = root_path / "src"
    if not (src_dir / "quant_fund").is_dir():
        src_dir = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else str(src_dir)

    # Input binding: declared tapes verify BEFORE argv runs — a drifted or
    # missing input means the replay could never reproduce the receipt, so
    # executing would only burn time (and feed a tampered tape to the lane's
    # parser). The proof records the attribution and fails closed.
    declared_tapes = manifest.get("input_tapes")
    input_tape_rows: list[dict[str, Any]] | None = None
    inputs_ok: bool | None = None
    if isinstance(declared_tapes, list):
        input_tape_rows, inputs_ok = _resolve_input_tapes(declared_tapes, root_path)

    exit_code: int | None = None
    timed_out = False
    spawn_error: str | None = None
    stderr_tail = ""
    inputs_blocked = inputs_ok is False
    # A declared artifact landing on a git-tracked path would clobber
    # committed evidence — fail before argv runs. Artifact digests are the
    # proof's only integrity surface; a tracked path means the bytes were
    # fixed at commit time and a hostile argv could overwrite them.
    committed_overwrite = _committed_artifact(manifest["artifacts"], root_path, run_cwd)
    started = time.monotonic()
    if inputs_blocked:
        spawn_error = "inputs_not_verified"
    elif committed_overwrite is not None:
        spawn_error = f"artifact_overwrites_committed:{committed_overwrite}"
    elif not run_cwd.is_dir():
        spawn_error = f"cwd_missing:{declared_cwd}"
    elif not run_cwd.is_relative_to(root_path):
        spawn_error = f"cwd_escapes_root:{declared_cwd}"
    else:
        # Delete declared artifacts first: a stale file must not satisfy a
        # digest pin when argv never produced it. Untracked deletions only —
        # committed paths were already refused by the overwrite guard.
        stale_error: str | None = None
        for entry in manifest["artifacts"]:
            declared = Path(str(entry["path"]))
            resolved = (declared if declared.is_absolute() else (run_cwd / declared)).resolve()
            if resolved.is_relative_to(root_path) and resolved.is_file():
                try:
                    resolved.unlink()
                except OSError as exc:
                    stale_error = f"artifact_stale_delete_failed:{entry['path']}:{exc}"
                    break
        if stale_error is not None:
            spawn_error = stale_error
        else:
            try:
                proc = subprocess.run(
                    resolved_argv,
                    cwd=run_cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_s),
                    check=False,
                )
                exit_code = int(proc.returncode)
                stderr_tail = (proc.stderr or "")[-_MAX_STDERR_TAIL:]
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                if exc.stderr:
                    tail = (
                        exc.stderr
                        if isinstance(exc.stderr, str)
                        else exc.stderr.decode("utf-8", errors="replace")
                    )
                    stderr_tail = tail[-_MAX_STDERR_TAIL:]
            except OSError as exc:
                spawn_error = f"{type(exc).__name__}: {exc}"
    elapsed_s = time.monotonic() - started

    artifact_rows: list[dict[str, Any]] = []
    for entry in manifest["artifacts"]:
        declared_path = str(entry["path"])
        expected = str(entry["sha256"])
        candidate = Path(declared_path)
        resolved = candidate if candidate.is_absolute() else (run_cwd / candidate)
        resolved = resolved.resolve()
        observed: str | None = None
        match = False
        note: str | None = None
        if not resolved.is_relative_to(root_path):
            note = "path_escapes_root"
        elif not resolved.is_file():
            note = "artifact_missing"
        else:
            observed = _file_sha256(resolved)
            match = observed == expected
        row: dict[str, Any] = {
            "path": declared_path,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "match": match,
        }
        if note is not None:
            row["note"] = note
        artifact_rows.append(row)

    all_match = (
        exit_code == 0
        and not timed_out
        and spawn_error is None
        and all(row["match"] for row in artifact_rows)
        and inputs_ok is not False
    )
    verdict = "pass" if all_match else "fail"
    body: dict[str, Any] = {
        "schema": REPLAY_PROOF_SCHEMA,
        "kind": "replay_proof",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": _source_data_label(path),
        "receipt": receipt_ref,
        "source_receipt_sha256": source_receipt_sha256,
        "argv": list(argv),
        "resolved_argv": resolved_argv,
        "artifacts": artifact_rows,
        "all_match": all_match,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed_s, 6),
        "timeout_s": float(timeout_s),
        "stderr_tail": stderr_tail,
        "verdict": verdict,
    }
    if spawn_error is not None:
        body["spawn_error"] = spawn_error
    if input_tape_rows is not None:
        body["input_tapes"] = input_tape_rows
        body["inputs_ok"] = bool(inputs_ok)
        if inputs_blocked:
            body["execution_skipped"] = "inputs_not_verified"
    if declared_cwd is not None:
        body["cwd"] = str(declared_cwd)
    if source_receipt_seal is not None:
        body["source_receipt_seal"] = source_receipt_seal
    if spawn_error is not None:
        body["spawn_error"] = spawn_error
    from quant_fund.research.receipt_v2 import seal_receipt

    # Seal the body itself so it verifies as a standalone v1 receipt and so
    # the receipt_sha256 key survives the receipt.v2 inner-seal check when
    # this body is wrapped in an envelope.
    return seal_receipt(body)


def replay_proof_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Internal consistency for a ``replay_proof.v1`` body — fail closed.

    Re-derives everything the sealed body claims about itself: field shape,
    digest formats, per-artifact ``match`` flags, and the headline
    ``all_match ⟺ (exit_code == 0 AND every artifact match)`` /
    ``verdict == ("pass" iff all_match)`` equivalences. A body that lies
    about its own arithmetic fails verification even under a fresh seal.
    """
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    errors: list[str] = []
    if payload.get("schema") != REPLAY_PROOF_SCHEMA:
        errors.append("schema")
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    data_label = payload.get("data_label")
    if not isinstance(data_label, str) or not data_label.strip():
        errors.append("data_label")
    receipt = payload.get("receipt")
    if not isinstance(receipt, str) or not receipt.strip():
        errors.append("receipt")
    if not _is_sha256(payload.get("source_receipt_sha256")):
        errors.append("source_receipt_sha256")
    source_receipt_seal = payload.get("source_receipt_seal")
    if source_receipt_seal is not None and not _is_sha256(source_receipt_seal):
        errors.append("source_receipt_seal")
    seal = payload.get("receipt_sha256")
    if not _is_sha256(seal):
        errors.append("receipt_sha256")
    else:
        # Re-derive the self-seal the same way the writers stamp it.
        from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

        unsealed = {key: value for key, value in payload.items() if key != "receipt_sha256"}
        try:
            if hash_bytes(canonical_json_bytes(unsealed)) != seal:
                errors.append("receipt_sha256")
        except (TypeError, ValueError):
            errors.append("receipt_sha256")
    argv = payload.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(arg, str) and arg for arg in argv)
    ):
        errors.append("argv")
    resolved_argv = payload.get("resolved_argv")
    if resolved_argv is not None and (
        not isinstance(resolved_argv, list)
        or not resolved_argv
        or not all(isinstance(arg, str) and arg for arg in resolved_argv)
    ):
        errors.append("resolved_argv")
    artifacts = payload.get("artifacts")
    artifact_matches: list[bool] = []
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts")
    else:
        for index, row in enumerate(artifacts):
            if not isinstance(row, Mapping):
                errors.append(f"artifacts[{index}]_not_object")
                continue
            art_path = row.get("path")
            if not isinstance(art_path, str) or not art_path.strip():
                errors.append(f"artifacts[{index}].path")
            expected = row.get("expected_sha256")
            observed = row.get("observed_sha256")
            if expected is not None and not _is_sha256(expected):
                errors.append(f"artifacts[{index}].expected_sha256")
            if observed is not None and not _is_sha256(observed):
                errors.append(f"artifacts[{index}].observed_sha256")
            match = row.get("match")
            if not isinstance(match, bool):
                errors.append(f"artifacts[{index}].match")
                continue
            # Recompute: a claimed match requires equal, present digests.
            want = (
                expected is not None
                and observed is not None
                and _is_sha256(expected)
                and expected == observed
            )
            if match != want:
                errors.append(f"artifacts[{index}].match")
            artifact_matches.append(match)
    input_tapes = payload.get("input_tapes")
    tape_matches: list[bool] = []
    inputs_ok_claimed = payload.get("inputs_ok")
    if input_tapes is not None:
        if not isinstance(input_tapes, list) or not input_tapes:
            errors.append("input_tapes")
        elif not isinstance(inputs_ok_claimed, bool):
            errors.append("inputs_ok")
        else:
            for index, row in enumerate(input_tapes):
                if not isinstance(row, Mapping):
                    errors.append(f"input_tapes[{index}]_not_object")
                    continue
                match = row.get("match")
                if not isinstance(match, bool):
                    errors.append(f"input_tapes[{index}].match")
                    continue
                expected = row.get("expected_sha256")
                observed = row.get("observed_sha256")
                if row.get("via_manifest") is None:
                    # Literal pin rows: match requires equal present digests.
                    want = (
                        expected is not None
                        and observed is not None
                        and _is_sha256(expected)
                        and _is_sha256(observed)
                        and expected == observed
                    )
                else:
                    want = bool(expected and observed and expected == observed)
                if match != want:
                    errors.append(f"input_tapes[{index}].match")
                tape_matches.append(match)
            # manifest-level rows carry match=False only (no digests) — the
            # row check above already enforces that.
            if inputs_ok_claimed != (bool(tape_matches) and all(tape_matches)):
                errors.append("inputs_ok")
    exit_code = payload.get("exit_code")
    if not (exit_code is None or (isinstance(exit_code, int) and not isinstance(exit_code, bool))):
        errors.append("exit_code")
    elapsed = payload.get("elapsed_s")
    if not (
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and float(elapsed) >= 0.0
    ):
        errors.append("elapsed_s")
    timeout_s = payload.get("timeout_s")
    if not (
        isinstance(timeout_s, (int, float))
        and not isinstance(timeout_s, bool)
        and math.isfinite(float(timeout_s))
        and float(timeout_s) > 0.0
    ):
        errors.append("timeout_s")
    timed_out = payload.get("timed_out")
    if not isinstance(timed_out, bool):
        errors.append("timed_out")
    all_match = payload.get("all_match")
    if not isinstance(all_match, bool):
        errors.append("all_match")
    else:
        want_all = (
            exit_code == 0
            and timed_out is False
            and bool(artifact_matches)
            and all(artifact_matches)
            and (input_tapes is None or inputs_ok_claimed is True)
        )
        if all_match != want_all:
            errors.append("all_match")
    if input_tapes is not None and inputs_ok_claimed is False:
        if payload.get("execution_skipped") != "inputs_not_verified":
            errors.append("execution_skipped")
        if exit_code is not None:
            errors.append("exit_code")
    verdict = payload.get("verdict")
    if (
        verdict not in REPLAY_PROOF_VERDICTS
        or isinstance(all_match, bool)
        and verdict != ("pass" if all_match else "fail")
    ):
        errors.append("verdict")
    return errors
