"""Replay proofs: re-execute a lane's declared argv and verify artifact bytes.

``verify-receipt`` proves a receipt's seal and re-derives claims *inside* the
payload; it never re-runs the lane. The honesty contract asks for more: every
research claim should be reproducible from a receipt hash. A **replay
manifest** closes that loop — a receipt may carry an optional ``replay``
block declaring the exact argv that produces its artifacts::

    "replay": {
        "argv": ["dipcatcher", "serial-watch", "tests/fixtures/replay/serial_pits.json",
                 "--data-label", "SYNTHETIC", "--out-dir", "data/metadata/replay"],
        "artifacts": [{"path": "data/metadata/replay/serial_watch_<d16>.json",
                       "sha256": "<64-hex file digest>"}],
        "cwd": "."                         # optional, relative to repo root
    }

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
    ``PATH``. Any other argv runs verbatim.
    """
    if argv and argv[0] in _CLI_ENTRYPOINTS:
        return [sys.executable, "-m", "quant_fund.cli.main", *argv[1:]]
    return list(argv)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    src_dir = root_path / "src"
    if (src_dir / "quant_fund").is_dir():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else str(src_dir)

    exit_code: int | None = None
    timed_out = False
    spawn_error: str | None = None
    stderr_tail = ""
    started = time.monotonic()
    if not run_cwd.is_dir():
        spawn_error = f"cwd_missing:{declared_cwd}"
    elif not run_cwd.is_relative_to(root_path):
        spawn_error = f"cwd_escapes_root:{declared_cwd}"
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
        )
        if all_match != want_all:
            errors.append("all_match")
    verdict = payload.get("verdict")
    if (
        verdict not in REPLAY_PROOF_VERDICTS
        or isinstance(all_match, bool)
        and verdict != ("pass" if all_match else "fail")
    ):
        errors.append("verdict")
    return errors
