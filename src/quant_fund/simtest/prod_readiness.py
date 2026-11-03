"""Checkout scorecard for institutional readiness and the operations runbook.

A row is ``pass`` only when this process observed the condition hold.
``needs-owner`` is an observation that the condition is outside this
checkout (a key holder, a vendor, a fleet host). It is not a pass.
The process exits nonzero when any row is ``fail``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from quant_fund.config.models import RuntimeConfig, RuntimeMode
from quant_fund.data.qualifying import resolve_feed
from quant_fund.execution.order_recon import LiveEndpointRefused
from quant_fund.paper.trading_session import PaperTradingSession, SessionLimits, run_session
from quant_fund.pipeline.e2e_smoke import require_finite, run_e2e_smoke
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent
from quant_fund.research.repo_integrity import verify_repo
from quant_fund.simtest.e2e_paper import run_e2e_paper
from quant_fund.simtest.paper_soak import run_paper_soak
from quant_fund.simtest.release_dry_run import run_release_dry_run
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.validation.live_readiness import live_readiness_report

_STATUSES = frozenset({"pass", "fail", "needs-owner"})
_CONTENTS_WRITE = re.compile(r"contents:\s*write")
_FX1_GUARD = (
    "hmac.compare_digest",
    "X-API-Key",
    "non-loopback binding requires FX1_API_KEY",
    "127.0.0.1",
)
_RESEARCH_APP_GUARD = (
    "credential_refusal",
    "X-API-Key",
    "QUANT_API_KEY",
    "127.0.0.1",
)
_FIXTURE_MARK = "not an fx-1 release candidate"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".ckpt")
_SECRET_MARKERS = ("ghp_", "ghs_", "github_pat_", "AKIA", "BEGIN PRIVATE")
_CI_CONTEXT = "cursor-ci"
_CI_SCHEMA = "cursor_ci_result.v1"
_CI_RESULT_RE = re.compile(r"CURSOR_CI_RESULT:\s*(pass|fail)\b")
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_GIT_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_CI_DOC_KEYS = frozenset({"schema", "context", "sha", "result"})


class ProdReadinessError(RuntimeError):
    """The scorecard itself was empty or tried to record a pass it did not observe."""


def run_prod_readiness(
    repo: str | Path,
    *,
    work: str | Path | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Run every check and return the receipt. Writes ``out`` when given."""
    root = Path(repo).resolve()
    if work is None:
        with tempfile.TemporaryDirectory(prefix="prod-readiness-") as tmp:
            return _finish(root, Path(tmp), out)
    return _finish(root, Path(work), out)


def main(argv: list[str] | None = None) -> int:
    """Print the table, write the receipt, and exit nonzero when a check fails."""
    parser = argparse.ArgumentParser(description="Production-readiness scorecard")
    parser.add_argument("--root", default=".", help="Repository checkout to score")
    parser.add_argument("--out", required=True, help="Path for the JSON receipt")
    parser.add_argument("--work", default=None, help="Writable directory for smoke artifacts")
    args = parser.parse_args(argv)
    try:
        receipt = run_prod_readiness(args.root, work=args.work, out=args.out)
    except ProdReadinessError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 1 if int(receipt["fail_count"]) else 0


def _finish(root: Path, work: Path, out: str | Path | None) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    checks = _collect(root, work)
    receipt = _receipt(checks)
    _print_table(receipt)
    if out is not None:
        destination = Path(out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            canonical_json_bytes(receipt).decode("utf-8") + "\n", encoding="utf-8"
        )
    return receipt


def _collect(root: Path, work: Path) -> list[dict[str, Any]]:
    checks = [
        _capture("python", lambda: _python()),
        _capture(
            "uv_lock", lambda: _command("uv_lock", ["uv", "lock", "--check"], cwd=root, timeout=120)
        ),
        _capture("ci_token_scope", lambda: _ci_token(root)),
        _capture(
            "api_auth_guard",
            lambda: _guard(root, "src/fx1/serve/api.py", _FX1_GUARD, "api_auth_guard"),
        ),
        _capture("research_api_auth_guard", lambda: _research_api_guard(root)),
        _capture("fx1_weights", lambda: _fx1_weights(root)),
        _capture("release_tag", lambda: _release_tags(root)),
        _capture("allow_live", lambda: _allow_live()),
        _capture("live_endpoint_refused", lambda: _live_endpoint(work)),
        _capture("qualifying_feed_closed", lambda: _qualifying_feed()),
    ]
    checks.extend(_capture_many("repo_integrity", lambda: _integrity(root)))
    checks.extend(
        [
            _capture("kill_switch", lambda: _kill_switch(work)),
            _capture("e2e_smoke", lambda: _e2e_smoke(work)),
            _capture("paper_session", lambda: _paper_session(work)),
            _capture("paper_soak", lambda: _paper_soak(work)),
            _capture("release_dry_run", lambda: _release_dry_run(work)),
            _capture(
                "ruff_check",
                lambda: _command(
                    "ruff_check", ["uv", "run", "ruff", "check", "."], cwd=root, timeout=180
                ),
            ),
            _capture(
                "ruff_format",
                lambda: _command(
                    "ruff_format",
                    ["uv", "run", "ruff", "format", "--check", "."],
                    cwd=root,
                    timeout=180,
                ),
            ),
            _capture(
                "mypy",
                lambda: _command(
                    "mypy", ["uv", "run", "mypy", "src/quant_fund"], cwd=root, timeout=600
                ),
            ),
            _capture(
                "doctor",
                lambda: _command(
                    "doctor",
                    ["uv", "run", "dipcatcher", "doctor", "--config", "configs/research.yaml"],
                    cwd=root,
                    timeout=180,
                ),
            ),
            _capture(
                "verify_research",
                lambda: _command(
                    "verify_research",
                    ["uv", "run", "dipcatcher", "verify-research"],
                    cwd=root,
                    timeout=180,
                ),
            ),
        ]
    )
    checks.extend(_owner_conditions())
    checks.append(
        _owner(
            "windows_fleet",
            "this checkout is not the Windows fleet host; WMI spawn and Defender exclusion are not observable here",
        )
    )
    checks.append(_capture("pr_test_suite", lambda: _pr_test_suite(root, work)))
    return checks


def _capture(check_id: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        row = fn()
    except Exception as exc:  # noqa: BLE001 — a check reports, it does not pass
        return _fail(check_id, f"{type(exc).__name__}: {exc}")
    if row.get("id") != check_id:
        return _fail(check_id, "check id mismatch")
    return row


def _capture_many(check_id: str, fn: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return [_fail(check_id, f"{type(exc).__name__}: {exc}")]


def _integrity(root: Path) -> list[dict[str, Any]]:
    report = verify_repo(root)
    gates = report["gates"]
    return [
        _crown(gates["crown_jewels"]),
        _pins(gates["pin_signatures"]),
        _epochs(gates),
        _anchors(gates["timestamp_anchors"]),
        _checkpoint(gates["checkpoint"]),
        _simple_signed("witness", gates["witness"], "witnessed", "witness proof verifies"),
        _simple_signed("checkpoint_spine", gates["spine"], "signed", "checkpoint spine verifies"),
        _rotations("key_rotation", gates["key_rotation"]),
        _rotations("quorum_rotations", gates["quorum_rotations"]),
        _lattice(gates["lattice"]),
    ]


def _crown(gate: Mapping[str, Any]) -> dict[str, Any]:
    errors = _errors(gate)
    if errors or gate.get("ok") is not True:
        shown = errors[0] if errors else "not ok"
        return _fail("crown_jewels", f"crown jewels drifted ({len(errors)}): {shown}")
    if gate.get("skipped"):
        return _owner("crown_jewels", "crown jewels were skipped")
    return _pass("crown_jewels", "crown jewel byte pins match this tree")


def _pins(gate: Mapping[str, Any]) -> dict[str, Any]:
    errors = _errors(gate)
    if gate.get("signed") is True and gate.get("ok") is True and not errors:
        return _pass("pin_signatures", "signed pin manifest verifies")
    if gate.get("signed") is not True:
        return _owner("pin_signatures", "pin manifest is not signed")
    shown = errors[0] if errors else "signature did not verify"
    return _fail("pin_signatures", f"signed pins do not verify: {shown}")


def _epochs(gates: Mapping[str, Any]) -> dict[str, Any]:
    epoch = {str(key): value for key, value in gates.items() if str(key).startswith("epoch:")}
    if not epoch:
        return _fail("epoch_chains", "no epoch chain was checked")
    failed: list[str] = []
    skipped: list[str] = []
    for name in sorted(epoch):
        gate = epoch[name]
        if not isinstance(gate, Mapping):
            failed.append(name)
            continue
        if gate.get("skipped"):
            skipped.append(name)
        elif _errors(gate) or gate.get("ok") is not True:
            failed.append(name)
    if failed:
        first = failed[0]
        gate = epoch[first]
        errors = _errors(gate) if isinstance(gate, Mapping) else ["malformed"]
        shown = errors[0] if errors else "not ok"
        return _fail(
            "epoch_chains",
            f"{len(failed)} of {len(epoch)} epoch chains are not stamped to this tree; {first} {shown}",
            failed=len(failed),
            corpora=len(epoch),
        )
    if skipped:
        return _owner(
            "epoch_chains",
            f"{len(skipped)} epoch chains are not verifiable on this checkout",
            skipped=len(skipped),
        )
    return _pass(
        "epoch_chains", f"{len(epoch)} epoch chains are stamped to this tree", corpora=len(epoch)
    )


def _anchors(gate: Mapping[str, Any]) -> dict[str, Any]:
    errors = _errors(gate)
    if errors or gate.get("ok") is not True:
        shown = errors[0] if errors else "not ok"
        return _fail("timestamp_anchors", f"timestamp anchors failed: {shown}")
    if gate.get("anchored") is not True:
        return _owner("timestamp_anchors", "timestamp anchors are absent")
    fresh = gate.get("fresh")
    if not isinstance(fresh, Mapping) or not fresh:
        return _fail("timestamp_anchors", "timestamp anchors are not fresh")
    stale = [str(name) for name, ok in fresh.items() if ok is not True]
    if stale:
        return _fail(
            "timestamp_anchors", f"timestamp anchors are not fresh for {stale[0]}", stale=len(stale)
        )
    return _pass(
        "timestamp_anchors", "timestamp anchors are fresh for every pinned file", files=len(fresh)
    )


def _checkpoint(gate: Mapping[str, Any]) -> dict[str, Any]:
    errors = _errors(gate)
    if errors or gate.get("ok") is not True or gate.get("signed") is not True:
        if gate.get("signed") is not True and not errors:
            return _owner("checkpoint", "checkpoint is not signed")
        shown = errors[0] if errors else "not ok"
        return _fail("checkpoint", f"checkpoint did not verify: {shown}")
    if gate.get("current") is True:
        return _pass("checkpoint", "signed checkpoint matches this tree")
    return _fail("checkpoint", "signed checkpoint is not current for this tree")


def _simple_signed(
    check_id: str, gate: Mapping[str, Any], flag: str, summary: str
) -> dict[str, Any]:
    errors = _errors(gate)
    if errors or gate.get("ok") is not True:
        shown = errors[0] if errors else "not ok"
        return _fail(check_id, f"{check_id} failed: {shown}")
    if gate.get(flag) is True:
        return _pass(check_id, summary)
    return _owner(check_id, f"{check_id} has no {flag} proof")


def _rotations(check_id: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    errors = _errors(gate)
    if errors or gate.get("ok") is not True:
        shown = errors[0] if errors else "not ok"
        return _fail(check_id, f"{check_id} failed: {shown}")
    count = int(gate.get("n_rotations") or 0)
    if count > 0:
        return _pass(check_id, f"{count} rotation records verify", count=count)
    return _owner(check_id, "verifier found no rotation records", count=0)


def _lattice(gate: Mapping[str, Any]) -> dict[str, Any]:
    errors = _errors(gate)
    verdict = str(gate.get("verdict") or "")
    if errors or gate.get("ok") is not True or verdict != "consistent":
        shown = errors[0] if errors else verdict or "not ok"
        return _fail("lattice", f"receipt lattice is {shown}")
    return _pass("lattice", "receipt lattice is consistent")


def _python() -> dict[str, Any]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # The scorecard still has to fail an older interpreter; this file can be
    # executed anywhere the project claims 3.12 is the floor.
    if sys.version_info >= (3, 12):  # noqa: UP036
        return _pass("python", f"python {version}", version=version)
    return _fail("python", f"python {version} is below 3.12", version=version)


def _ci_token(root: Path) -> dict[str, Any]:
    workflow_dir = root / ".github" / "workflows"
    files = sorted(workflow_dir.glob("*.yml")) if workflow_dir.is_dir() else []
    if not files:
        return _fail("ci_token_scope", "no workflow files")
    writes = [
        path.name for path in files if _CONTENTS_WRITE.search(path.read_text(encoding="utf-8"))
    ]
    if writes:
        return _fail(
            "ci_token_scope", f"workflow grants contents write: {writes[0]}", files=len(files)
        )
    return _pass(
        "ci_token_scope", f"{len(files)} workflows grant no contents write", files=len(files)
    )


def _research_api_guard(root: Path) -> dict[str, Any]:
    """The lab API refuses credentials in app.py and compares them in http_controls."""
    app = _guard(
        root,
        "src/quant_fund/api/app.py",
        _RESEARCH_APP_GUARD,
        "research_api_auth_guard",
    )
    if app["status"] != "pass":
        return app
    compare = root / "src/quant_fund/api/http_controls.py"
    marker = "hmac.compare_digest"
    if not compare.is_file() or marker not in compare.read_text(encoding="utf-8"):
        return _fail(
            "research_api_auth_guard",
            f"missing guard marker in src/quant_fund/api/http_controls.py: {marker}",
        )
    return _pass(
        "research_api_auth_guard",
        "auth guard markers present in src/quant_fund/api/app.py",
    )


def _guard(root: Path, relative: str, markers: Sequence[str], check_id: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return _fail(check_id, f"missing {relative}")
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        return _fail(check_id, f"missing guard marker in {relative}: {missing[0]}")
    return _pass(check_id, f"auth guard markers present in {relative}", bytes=path.stat().st_size)


def _fx1_weights(root: Path) -> dict[str, Any]:
    counted = 0
    fixture = False
    for base in (root / "artifacts", root / "data" / "models", root / "models"):
        if not base.is_dir():
            continue
        for card in base.rglob("modelcard.json"):
            found, is_fixture = _weight_card(card)
            fixture = fixture or is_fixture
            counted += int(found)
    if counted:
        return _pass("fx1_weights", f"{counted} published fx-1 weight cards", count=counted)
    return _fail(
        "fx1_weights",
        "no published fx-1 weights; the tiny LM fixture is not a release",
        fixture_excluded=fixture,
        count=0,
    )


def _weight_card(card: Path) -> tuple[bool, bool]:
    if "kronos" in card.as_posix().lower() or "fx1_tiny_lm" in card.as_posix():
        return False, "fx1_tiny_lm" in card.as_posix()
    try:
        payload = json.loads(card.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProdReadinessError(f"unreadable model card {card.name}") from exc
    if not isinstance(payload, dict):
        return False, False
    limits = " ".join(str(item) for item in payload.get("known_limits") or [])
    if _FIXTURE_MARK in limits:
        return False, True
    identity = " ".join(str(payload.get(key, "")) for key in ("version", "base_model"))
    if "fx-1" not in identity.lower() and "fx1" not in identity.lower():
        return False, False
    return _has_weight_file(card.parent), False


def _has_weight_file(directory: Path) -> bool:
    return any(path.suffix in _WEIGHT_SUFFIXES and path.is_file() for path in directory.iterdir())


def _release_tags(root: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["git", "-C", str(root), "tag", "-l", "v*"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        return _fail("release_tag", f"git tag failed: {_tail(proc.stderr)}")
    tags = [line for line in proc.stdout.splitlines() if line.startswith("v")]
    if not tags:
        return _fail("release_tag", "no v* release tag is present", count=0)
    return _pass("release_tag", f"{len(tags)} v* release tags", count=len(tags))


def _allow_live() -> dict[str, Any]:
    try:
        RuntimeConfig(mode=RuntimeMode.LIVE, allow_live=True)
    except ValueError as exc:
        if "unsupported" in str(exc):
            return _pass("allow_live", "runtime.allow_live is refused")
        return _fail("allow_live", str(exc))
    return _fail("allow_live", "runtime.allow_live was accepted")


def _live_endpoint(work: Path) -> dict[str, Any]:
    try:
        PaperTradingSession(state_path=work / "live-endpoint.json", endpoint_kind="ibkr")
    except LiveEndpointRefused:
        return _pass("live_endpoint_refused", "a live endpoint is refused")
    return _fail("live_endpoint_refused", "a live endpoint was accepted")


def _qualifying_feed() -> dict[str, Any]:
    feed = resolve_feed(entitled=False)
    closed = (
        feed.get("qualifying") is False
        and feed.get("synthetic_substitution") is False
        and feed.get("status") == "UNAVAILABLE"
        and feed.get("live_pnl_claim") is False
    )
    if not closed:
        return _fail("qualifying_feed_closed", "missing entitlement did not stay unavailable")
    return _pass(
        "qualifying_feed_closed", "an unentitled feed stays unavailable with no synthetic fill-in"
    )


def _pr_test_suite(root: Path, work: Path) -> dict[str, Any]:
    """Read an external cursor-ci attestation. This function does not sign or post one."""
    sha = _head_sha(root)
    if sha is None:
        return _owner("pr_test_suite", "this checkout has no commit sha to attest")
    status_word = _status_channel(root, sha)
    file_word = _file_channel(root, work, sha)
    return _ci_attestation_row(status_word, file_word, sha)


def _ci_attestation_row(status_word: str | None, file_word: str | None, sha: str) -> dict[str, Any]:
    words = [word for word in (status_word, file_word) if word in {"pass", "fail"}]
    broken = file_word in {"bad-signature", "malformed"}
    if "fail" in words or broken:
        if "pass" in words or broken:
            return _fail(
                "pr_test_suite",
                "cursor-ci attestation is not a clean pass for this sha",
                sha=sha,
            )
        return _fail("pr_test_suite", "cursor-ci reported failure for this sha", sha=sha)
    if "pass" in words:
        return _pass("pr_test_suite", "cursor-ci attested the full suite for this sha", sha=sha)
    return _owner(
        "pr_test_suite",
        "the full pytest suite belongs to the CI test job; no cursor-ci attestation for this sha",
        sha=sha,
    )


def _status_channel(root: Path, sha: str) -> str | None:
    payload = _load_commit_status(root, sha)
    if not isinstance(payload, dict):
        return None
    reported = payload.get("sha")
    if isinstance(reported, str) and reported != sha:
        return None
    return _judge_status_payload(payload)


def _judge_status_payload(payload: Mapping[str, Any]) -> str | None:
    entries = payload.get("statuses")
    if not isinstance(entries, list):
        return None
    matches = [
        item for item in entries if isinstance(item, dict) and item.get("context") == _CI_CONTEXT
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""))
    chosen = matches[-1]
    state = chosen.get("state")
    description = chosen.get("description")
    token = _one_result_token(description if isinstance(description, str) else None)
    if not isinstance(state, str):
        return "fail"
    if state == "success" and token == "pass":
        return "pass"
    if state == "failure" and token == "fail":
        return "fail"
    if state == "pending" and token is None:
        return None
    return "fail"


def _one_result_token(text: str | None) -> str | None:
    if not text:
        return None
    found = _CI_RESULT_RE.findall(text)
    if len(found) != 1:
        return None
    if found[0] == "pass":
        return "pass"
    if found[0] == "fail":
        return "fail"
    return None


def _file_channel(root: Path, work: Path, sha: str) -> str | None:
    raw = os.environ.get("CURSOR_CI_RESULT_FILE", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file() or _refused_result_path(root, work, path):
        return None
    try:
        public = _attestation_pubkey()
    except ValueError:
        return "malformed"
    if public is None:
        return None
    signature_path = _signature_path(path)
    if signature_path is None or _refused_result_path(root, work, signature_path):
        return "bad-signature"
    try:
        body = path.read_bytes()
        signature = bytes.fromhex(signature_path.read_text(encoding="utf-8").strip())
        document = json.loads(body)
    except (OSError, ValueError, json.JSONDecodeError):
        return "malformed"
    if not _ed25519_verify(public, signature, body):
        return "bad-signature"
    return _judge_result_document(document, sha)


def _signature_path(result: Path) -> Path | None:
    raw = os.environ.get("CURSOR_CI_RESULT_SIG", "").strip()
    path = Path(raw) if raw else result.with_name(result.name + ".sig")
    if path.is_file():
        return path
    return None


def _judge_result_document(document: object, sha: str) -> str | None:
    if not isinstance(document, dict) or set(document) != _CI_DOC_KEYS:
        return "malformed"
    if document.get("schema") != _CI_SCHEMA or document.get("context") != _CI_CONTEXT:
        return "malformed"
    if document.get("sha") != sha:
        return None
    result = document.get("result")
    if result == "pass":
        return "pass"
    if result == "fail":
        return "fail"
    return "malformed"


def _ed25519_verify(public: bytes, signature: bytes, message: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


def _attestation_pubkey() -> bytes | None:
    hex_key = os.environ.get("CURSOR_CI_ATTESTATION_PUBKEY", "").strip()
    file_key = os.environ.get("CURSOR_CI_ATTESTATION_PUBKEY_FILE", "").strip()
    if hex_key:
        return _raw_public_key(hex_key)
    if not file_key:
        return None
    text = Path(file_key).read_text(encoding="utf-8")
    if any(marker in text for marker in _SECRET_MARKERS):
        raise ValueError("attestation public key is not a public key")
    return _raw_public_key(text.strip())


def _raw_public_key(text: str) -> bytes:
    try:
        key = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError("attestation public key is malformed") from exc
    if len(key) != 32:
        raise ValueError("attestation public key is malformed")
    return key


def _refused_result_path(root: Path, work: Path, path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        resolved = path.resolve()
        work_resolved = work.resolve()
        root_resolved = root.resolve()
    except OSError:
        return True
    if resolved == work_resolved or work_resolved in resolved.parents:
        return True
    if resolved == root_resolved or root_resolved not in resolved.parents:
        return False
    relative = resolved.relative_to(root_resolved).as_posix()
    return _tracked(root, relative)


def _tracked(root: Path, relative: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _head_sha(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    sha = proc.stdout.strip()
    if proc.returncode == 0 and _SHA_RE.fullmatch(sha):
        return sha
    return None


def _load_commit_status(root: Path, sha: str) -> object | None:
    slug = _github_slug(root)
    if slug is None:
        return None
    token = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if token:
        return _fetch_status(slug, sha, token)
    return _fetch_status_gh(slug, sha)


def _decoded_json(raw: bytes | str) -> object | None:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed


def _fetch_status(slug: str, sha: str, token: str) -> object | None:
    from quant_fund.utils.net_timeout import get_capped

    try:
        raw = get_capped(
            f"https://api.github.com/repos/{slug}/commits/{sha}/status",
            {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "dipcatcher-prod-readiness",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10,
            max_bytes=1_000_000,
        )
    except Exception:  # noqa: BLE001 — a missing status is needs-owner, and the error can carry a token
        return None
    return _decoded_json(raw)


def _fetch_status_gh(slug: str, sha: str) -> object | None:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GH_PROMPT_DISABLED"] = "1"
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{slug}/commits/{sha}/status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return _decoded_json(proc.stdout)


def _github_slug(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return _slug_from_remote(proc.stdout.strip())


def _slug_from_remote(url: str) -> str | None:
    text = url.strip()
    if text.endswith(".git"):
        text = text[:-4]
    marker = "github.com"
    index = text.rfind(marker)
    if index < 0:
        return None
    tail = text[index + len(marker) :]
    if tail.startswith(":") or tail.startswith("/"):
        tail = tail[1:]
    else:
        return None
    parts = [part for part in tail.split("/") if part]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if _GIT_NAME_RE.fullmatch(owner) is None or _GIT_NAME_RE.fullmatch(repo) is None:
        return None
    return f"{owner}/{repo}"


def _kill_switch(work: Path) -> dict[str, Any]:
    limits = SessionLimits(
        max_order_quantity=10.0,
        max_order_notional=1_000_000.0,
        max_position_notional=1_000_000.0,
        max_gross_notional=1_000_000.0,
    )
    report = run_session(
        state_path=work / "kill.json",
        commands=[
            {"op": "mark", "security_id": "SEC_0001", "price": 10.0},
            {
                "op": "submit",
                "order_id": "open",
                "security_id": "SEC_0001",
                "side": "BUY",
                "quantity": 1,
            },
            {"op": "kill", "reason": "readiness"},
            {
                "op": "submit",
                "order_id": "halted",
                "security_id": "SEC_0001",
                "side": "BUY",
                "quantity": 1,
            },
            {"op": "cancel", "order_id": "open"},
        ],
        starting_cash=1_000.0,
        limits=limits,
    )
    steps = report["steps"]
    halted = steps[3]
    cancelled = steps[4]
    held = (
        len(steps) == 5
        and halted["verdict"] == "rejected"
        and "kill switch" in str(halted.get("reason", ""))
        and cancelled["verdict"] == "cancelled"
        and report["live_orders_enabled"] is False
        and report["would_promote_live"] is False
        and report["live_pnl_claim"] is False
    )
    if not held:
        return _fail("kill_switch", "kill switch did not halt new orders while leaving cancel open")
    return _pass("kill_switch", "new orders halt and cancel still succeeds")


def _e2e_smoke(work: Path) -> dict[str, Any]:
    first = run_e2e_smoke(work / "smoke-a")
    second = run_e2e_smoke(work / "smoke-b")
    if (
        first != second
        or first.get("data_label") != "SYNTHETIC"
        or first.get("live_pnl_claim") is not False
    ):
        return _fail("e2e_smoke", "two smoke runs did not share a SYNTHETIC receipt")
    digest = str(first.get("digest") or "")
    if len(digest) != 64:
        return _fail("e2e_smoke", "smoke digest is empty")
    return _pass("e2e_smoke", "two smoke runs shared a digest", digest=digest)


def _paper_session(work: Path) -> dict[str, Any]:
    first = run_e2e_paper(work / "paper-a")
    second = run_e2e_paper(work / "paper-b")
    controls = first.get("controls")
    if not isinstance(controls, dict):
        return _fail("paper_session", "paper receipt has no controls")
    held = (
        first == second
        and first.get("data_label") == "SYNTHETIC"
        and first.get("live_pnl_claim") is False
        and controls.get("oversize_reason") == "max_order_quantity"
        and controls.get("halted_submit") is True
        and controls.get("cancel_verdict") == "cancelled"
        and controls.get("restart_match") is True
    )
    if not held:
        return _fail("paper_session", "paper session controls did not hold on two runs")
    return _pass("paper_session", "two paper sessions shared kill, cancel, oversize, and restart")


def _paper_soak(work: Path) -> dict[str, Any]:
    first = run_paper_soak(work / "soak-a")
    second = run_paper_soak(work / "soak-b")
    held = (
        first == second
        and int(first["sessions"]) >= 30
        and int(first["days_checked"]) >= 30
        and float(first["max_cash_gap"]) <= 1e-6
        and float(first["max_position_gap"]) <= 1e-9
        and float(first["max_trade_residual"]) <= 1e-6
        and float(first["max_mark_residual"]) <= 1e-6
        and first.get("live_pnl_claim") is False
        and first.get("data_label") == "SYNTHETIC"
    )
    if not held:
        return _fail("paper_soak", "soak accounting did not hold for every session")
    return _pass(
        "paper_soak", f"{first['sessions']} synthetic sessions conserved cash and positions"
    )


def _release_dry_run(work: Path) -> dict[str, Any]:
    report = run_release_dry_run(work / "wheel")
    entrypoints = report["entrypoints"]
    held = (
        str(report["wheel"]).endswith(".whl")
        and str(report["sdist"]).endswith(".tar.gz")
        and int(entrypoints["dipcatcher"]["help_bytes"]) > 0
        and int(entrypoints["fx1"]["help_bytes"]) > 0
        and int(entrypoints["dipcatcher"]["exit_code"]) == 0
        and int(entrypoints["fx1"]["exit_code"]) == 0
    )
    if not held:
        return _fail("release_dry_run", "built CLI entrypoints did not print help")
    return _pass("release_dry_run", "wheel and sdist built; dipcatcher and fx1 --help exited 0")


def _owner_conditions() -> list[dict[str, Any]]:
    report = live_readiness_report()
    closed = (
        report.get("eligible") is False
        and report.get("would_promote_live") is False
        and report.get("live_pnl_claim") is False
        and report.get("live_orders_enabled") is False
    )
    rows = [
        _pass("live_block", "harness reports live capital ineligible")
        if closed
        else _fail("live_block", "live block is not closed")
    ]
    for condition in report["conditions"]:
        rows.append(_owner_condition(condition, eligible=report.get("eligible") is True))
    return rows


def _owner_condition(condition: Mapping[str, Any], *, eligible: bool) -> dict[str, Any]:
    status = str(condition.get("status") or "")
    name = str(condition.get("name") or "condition")
    check_id = f"live_condition_{condition.get('id')}"
    satisfied = status.strip().upper() in {"PASS", "READY", "SATISFIED", "OK"}
    if eligible or satisfied:
        return _fail(check_id, f"{name} is marked {status or 'eligible'} without external evidence")
    return _owner(check_id, f"{name}: {status}", reported_status=status)


def _command(check_id: str, argv: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _fail(check_id, f"timed out after {timeout}s")
    if proc.returncode == 0:
        return _pass(check_id, "exit 0", exit_code=0)
    return _fail(
        check_id,
        f"exit {proc.returncode}: {_tail(proc.stderr or proc.stdout)}",
        exit_code=proc.returncode,
    )


def _receipt(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not checks:
        raise ProdReadinessError("prod-readiness produced no checks")
    ids = [str(check["id"]) for check in checks]
    if len(ids) != len(set(ids)):
        raise ProdReadinessError("prod-readiness produced a duplicate check")
    counts = {status: sum(check["status"] == status for check in checks) for status in _STATUSES}
    ready = counts["pass"] == len(checks)
    body: dict[str, Any] = {
        "schema": "prod_readiness.v1",
        "data_label": "OPERATIONS",
        "research_only": True,
        "live_pnl_claim": False,
        "live_orders_enabled": False,
        "would_promote_live": False,
        "ready": ready,
        "pass_count": counts["pass"],
        "fail_count": counts["fail"],
        "needs_owner_count": counts["needs-owner"],
        "checks": [dict(check) for check in checks],
    }
    require_finite("prod-readiness", body)
    if not family_blob_forbidden_metrics_absent(body):
        raise ProdReadinessError("prod-readiness receipt used a forbidden headline key")
    return {**body, "digest": hash_bytes(canonical_json_bytes(body))}


def _print_table(receipt: Mapping[str, Any]) -> None:
    print(f"{'status':<12} {'check':<28} summary")
    for check in receipt["checks"]:
        print(f"{check['status']:<12} {check['id']:<28} {check['summary']}")
    print(
        f"pass={receipt['pass_count']} fail={receipt['fail_count']} "
        f"needs-owner={receipt['needs_owner_count']} ready={'yes' if receipt['ready'] else 'no'}"
    )


def _row(check_id: str, status: str, summary: str, **fields: Any) -> dict[str, Any]:
    if status not in _STATUSES:
        raise ProdReadinessError(f"{check_id}: unknown status {status}")
    if "ok" in fields and fields["ok"] is not (status == "pass"):
        raise ProdReadinessError(f"{check_id}: status {status} disagrees with the observation")
    if fields.get("observed") is False:
        raise ProdReadinessError(f"{check_id}: pass without an observation")
    evidence: dict[str, Any] = {"observed": True, **fields, "ok": status == "pass"}
    return {
        "id": check_id,
        "status": status,
        "summary": " ".join(summary.split()),
        "evidence": evidence,
    }


def _pass(check_id: str, summary: str, **fields: Any) -> dict[str, Any]:
    return _row(check_id, "pass", summary, **fields)


def _fail(check_id: str, summary: str, **fields: Any) -> dict[str, Any]:
    return _row(check_id, "fail", summary, **fields)


def _owner(check_id: str, summary: str, **fields: Any) -> dict[str, Any]:
    return _row(check_id, "needs-owner", summary, **fields)


def _errors(gate: Mapping[str, Any]) -> list[str]:
    raw = gate.get("errors") or []
    if not isinstance(raw, list):
        return ["malformed errors"]
    return [str(item) for item in raw]


def _tail(text: str) -> str:
    raw = text or ""
    if any(marker in raw for marker in _SECRET_MARKERS):
        return "output redacted"
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    chosen = next((line for line in lines if _finding(line)), "")
    if not chosen and lines:
        chosen = lines[-1]
    chosen = " ".join(chosen.split())
    return chosen[:160] or "no output"


def _finding(line: str) -> bool:
    folded = line.lower()
    if "invalid" not in folded and "error" not in folded:
        return False
    token = folded.strip("\"',:[]{} ")
    return token not in {"error", "errors"}


if __name__ == "__main__":
    raise SystemExit(main())
