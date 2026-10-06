"""Deployment doctor — structured diagnosis for a running (or would-be)
fx-1 harness deployment.

``selftest`` is the golden-path functional gate: it proves a request can
traverse the stack. ``doctor`` answers the different question — *why* is
this deployment broken or degraded — by introspecting what the process
already knows: which backends are configured, whether the BYOK endpoint
answers an auth probe, whether the checkpoint loads, whether journals
replay, whether any credential can still authenticate, and whether the
request pool is saturated or drained. Every check is read-only — nothing
is exercised, mutated, or billed.

One builder feeds all legs: ``GET /harness/doctor`` (admin-scoped) calls
it with the app's live internals; ``Fx1Harness.doctor`` calls it against
the in-process stores; ``HarnessClient.doctor`` proxies the route and
appends the wire-side checks only a client can see (api-version contract,
OpenAPI reachability); ``fx1 harness doctor`` renders the report and maps
the verdict onto the pinned exit codes — ``0`` healthy, ``1`` broken,
``2`` degraded.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

import fx1.serve.backends as _be
from fx1 import __version__
from fx1.serve.contract import API_VERSION
from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore

__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "build_doctor_report",
    "doctor_verdict",
]

# Disk-free floor for the state-dir warning — below this the journals can
# no longer be trusted to complete an append.
_DISK_WARN_BYTES = 64 * 1024 * 1024
_DISK_WARN_FRACTION = 0.05

# Doctor probes share the credentialed BYOK wire seam so the private-
# network policy (``FX1_BYOK_ALLOW_PRIVATE_NETWORKS``) binds them exactly
# like a real completion call.
_DOCTOR_PROBE_TIMEOUT_S = 5.0


class DoctorCheck(BaseModel):
    """One probe's verdict.

    ``severity`` is the failure's weight when ``ok`` is false: ``error``
    marks the deployment broken, ``warn`` only degrades it. A check that
    does not apply (feature unconfigured) reports ``ok: true`` with a
    "not configured" detail rather than disappearing — the check list is
    stable so dashboards can diff deployments.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    ok: bool
    detail: str = ""
    severity: Literal["error", "warn"] = "error"


class DoctorReport(BaseModel):
    """The deployment's diagnosis. ``verdict`` is derived from the checks
    at validation time — revalidate after mutating ``checks``."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["server", "in_process", "remote"]
    checked_at: float
    verdict: Literal["healthy", "degraded", "broken"]
    checks: list[DoctorCheck]

    @property
    def ok(self) -> bool:
        return self.verdict == "healthy"


def doctor_verdict(
    checks: Iterable[DoctorCheck],
) -> Literal["healthy", "degraded", "broken"]:
    """Reduce a check list to the verdict — broken on any failed
    ``error``-severity check, degraded on failed ``warn`` only."""
    if any(not c.ok and c.severity == "error" for c in checks):
        return "broken"
    if any(not c.ok for c in checks):
        return "degraded"
    return "healthy"


def _note(
    checks: list[DoctorCheck],
    name: str,
    ok: bool,
    detail: str = "",
    *,
    severity: Literal["error", "warn"] = "error",
) -> None:
    checks.append(DoctorCheck(name=name, ok=ok, detail=detail, severity=severity))


class _OpsSnapshot(Protocol):
    """The fields doctor reads off ``metrics.snapshot()`` — mirrors
    ``MetricsResponse`` without importing it (api.py imports doctor)."""

    uptime_s: float
    requests_total: int
    errors_total: int
    inflight: int
    inflight_watermark: int
    max_inflight: int
    draining: bool
    rate_limited_total: int


class _OpsMetrics(Protocol):
    """The server's request counters + inflight gauge + drain latch."""

    @property
    def draining(self) -> threading.Event: ...

    def snapshot(self) -> _OpsSnapshot: ...


def _check_backends(checks: list[DoctorCheck], *, probe_backends: bool, timeout_s: float) -> None:
    byok_url = os.environ.get(_be.BYOK_BASE_URL_ENV, "").strip()
    byok_key = os.environ.get(_be.BYOK_API_KEY_ENV, "")
    byok_model = os.environ.get(_be.BYOK_MODEL_ENV, "")
    checkpoint = os.environ.get("FX1_CHECKPOINT_DIR", "").strip()
    serve_url = os.environ.get(_be.LOCAL_SERVE_URL_ENV, "").strip()
    serve_cmd = os.environ.get(_be.LOCAL_SERVE_CMD_ENV, "").strip()
    hosted = bool(os.environ.get("MOONSHOT_API_KEY", ""))

    byok_configured = bool(byok_url and byok_key and byok_model)
    local_configured = bool(
        checkpoint and (Path(checkpoint) / "modelcard.json").is_file() and (serve_url or serve_cmd)
    )
    names = [
        name
        for name, on in (
            ("local_fx1", local_configured),
            ("byok", byok_configured),
            ("hosted_k3", hosted),
        )
        if on
    ]
    _note(
        checks,
        "config.backends",
        bool(names),
        "configured: " + ", ".join(names) if names else "no backend configured",
    )

    _check_byok(
        checks,
        byok_url=byok_url,
        byok_key=byok_key,
        byok_model=byok_model,
        probe_backends=probe_backends,
        timeout_s=timeout_s,
    )
    _check_local_engine(
        checks,
        checkpoint=checkpoint,
        serve_url=serve_url,
        serve_cmd=serve_cmd,
        probe_backends=probe_backends,
    )
    _note(
        checks,
        "config.hosted_k3",
        True,
        "MOONSHOT_API_KEY set" if hosted else "MOONSHOT_API_KEY unset",
    )


def _check_byok(
    checks: list[DoctorCheck],
    *,
    byok_url: str,
    byok_key: str,
    byok_model: str,
    probe_backends: bool,
    timeout_s: float,
) -> None:
    """BYOK leg: env completeness → URL grammar → one cheap GET on the
    ``models`` sibling of the chat route — auth probe, never a paid
    completion."""
    present = [
        name
        for name, val in (
            (_be.BYOK_BASE_URL_ENV, byok_url),
            (_be.BYOK_API_KEY_ENV, byok_key),
            (_be.BYOK_MODEL_ENV, byok_model),
        )
        if val
    ]
    if not present:
        _note(checks, "config.byok", True, "not configured")
        return
    if len(present) < 3:
        missing = [
            name
            for name, val in (
                (_be.BYOK_BASE_URL_ENV, byok_url),
                (_be.BYOK_API_KEY_ENV, byok_key),
                (_be.BYOK_MODEL_ENV, byok_model),
            )
            if not val
        ]
        _note(
            checks,
            "config.byok",
            False,
            "partially configured — missing " + ", ".join(missing),
        )
        return
    problem = _be.byok_base_url_problem(byok_url)
    if problem is not None:
        # byok_base_url_problem never echoes the URL into the reason.
        _note(checks, "config.byok", False, f"FX1_BYOK_BASE_URL {problem}")
        return
    if not probe_backends:
        _note(checks, "config.byok", True, "configured (endpoint probe skipped)")
        return
    models_url = _be._openai_sibling_url(_be._chat_completions_url(byok_url), "models")
    request = urllib.request.Request(models_url, method="GET")
    request.add_header("Authorization", f"Bearer {byok_key}")
    _be._apply_byok_destination_policy(request)
    try:
        with _be._openai_urlopen(request, timeout_s=timeout_s):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            _note(
                checks,
                "config.byok",
                False,
                f"endpoint answered HTTP {exc.code} — credential refused",
            )
        else:
            _note(
                checks,
                "config.byok",
                False,
                f"endpoint reachable but /models answered HTTP {exc.code}",
                severity="warn",
            )
        return
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        _note(checks, "config.byok", False, f"endpoint unreachable: {reason}")
        return
    _note(checks, "config.byok", True, "endpoint reachable — credential accepted")


def _check_local_engine(
    checks: list[DoctorCheck],
    *,
    checkpoint: str,
    serve_url: str,
    serve_cmd: str,
    probe_backends: bool,
) -> None:
    """local_fx1 leg: checkpoint present + card loadable + engine link —
    the same gates ``LocalFx1Backend.__init__`` enforces, read-only."""
    if not checkpoint and not serve_url and not serve_cmd:
        _note(checks, "config.local_fx1", True, "not configured")
        return
    root = Path(checkpoint) if checkpoint else None
    if root is None or not root.is_dir():
        _note(
            checks,
            "config.local_fx1",
            False,
            "FX1_CHECKPOINT_DIR missing or not a directory",
        )
        return
    card_path = root / "modelcard.json"
    if not card_path.is_file():
        _note(checks, "config.local_fx1", False, f"no model card at {card_path}")
        return
    from fx1.modelcard import ModelCard  # noqa: PLC0415 — heavy import, deferred

    try:
        card = ModelCard.load(card_path)
    except Exception as exc:  # noqa: BLE001 — report the parse fault, not a traceback
        _note(checks, "config.local_fx1", False, f"model card unreadable: {exc}")
        return
    if not card.eval_delta.ship_eligible:
        _note(
            checks,
            "config.local_fx1",
            False,
            f"{card.version} failed the ship gate — refusing to serve",
        )
        return
    if os.environ.get("FX1_SIGNING_KEY"):
        from fx1.serve.signing import verify_release  # noqa: PLC0415

        try:
            signed = verify_release(root)
        except Exception as exc:  # noqa: BLE001 — a verifier fault is a broken deploy
            _note(
                checks,
                "config.local_fx1",
                False,
                f"release signature check errored: {exc}",
            )
            return
        if not signed:
            _note(
                checks,
                "config.local_fx1",
                False,
                "release signature missing or invalid under FX1_SIGNING_KEY",
            )
            return
    if not serve_url and not serve_cmd:
        _note(
            checks,
            "config.local_fx1",
            False,
            "checkpoint ok but no engine link — set FX1_LOCAL_SERVE_URL or FX1_LOCAL_SERVE_CMD",
        )
        return
    if serve_url:
        parsed = urllib.parse.urlparse(serve_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            _note(
                checks,
                "config.local_fx1",
                False,
                f"{_be.LOCAL_SERVE_URL_ENV} must be an http(s) URL",
            )
            return
        if not probe_backends:
            _note(checks, "config.local_fx1", True, "checkpoint ok (engine probe skipped)")
            return
        probe = f"{parsed.scheme}://{parsed.netloc}/v1/models"
        try:
            # Same probe LocalFx1Backend._engine_up runs — raw urllib, the
            # operator-declared local engine, no credentials attached.
            with urllib.request.urlopen(probe, timeout=1):  # noqa: S310  # nosec B310
                engine_up = True
        except urllib.error.HTTPError:
            engine_up = True  # any HTTP answer means a server is listening
        except (urllib.error.URLError, OSError):
            engine_up = False
        if engine_up:
            _note(checks, "config.local_fx1", True, "checkpoint ok — engine answering")
        elif serve_cmd:
            _note(
                checks,
                "config.local_fx1",
                False,
                "engine not answering; FX1_LOCAL_SERVE_CMD will cold-start it",
                severity="warn",
            )
        else:
            _note(
                checks,
                "config.local_fx1",
                False,
                "engine unreachable and no FX1_LOCAL_SERVE_CMD to spawn it",
            )
        return
    _note(
        checks,
        "config.local_fx1",
        True,
        "checkpoint ok — engine spawned lazily via FX1_LOCAL_SERVE_CMD",
    )


def _check_state_dir(checks: list[DoctorCheck], state_path: Path | None) -> None:
    if state_path is None:
        _note(checks, "state.dir", True, "in-memory only — no durable state dir")
        _note(checks, "state.disk", True, "skipped (in-memory)")
        return
    if state_path.exists():
        if not state_path.is_dir():
            _note(
                checks,
                "state.dir",
                False,
                f"{state_path} exists but is not a directory",
            )
        elif os.access(state_path, os.W_OK | os.X_OK):
            _note(checks, "state.dir", True, f"{state_path} writable")
        else:
            _note(checks, "state.dir", False, f"{state_path} is not writable")
    else:
        # Journal appends mkdir -p lazily — a missing dir is fine iff some
        # ancestor can be written into.
        ancestor = state_path.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if ancestor.exists() and os.access(ancestor, os.W_OK | os.X_OK):
            _note(
                checks,
                "state.dir",
                True,
                f"{state_path} does not exist yet (created on first journal write)",
            )
        else:
            _note(
                checks,
                "state.dir",
                False,
                f"{state_path} does not exist and cannot be created",
            )
    try:
        usage = shutil.disk_usage(state_path if state_path.exists() else ancestor)
    except OSError as exc:
        _note(checks, "state.disk", False, f"cannot stat filesystem: {exc}")
        return
    free_pct = usage.free / usage.total * 100 if usage.total else 0.0
    detail = f"{usage.free / 2**30:.2f} GiB free ({free_pct:.1f}%)"
    if usage.free < _DISK_WARN_BYTES or free_pct < _DISK_WARN_FRACTION * 100:
        _note(checks, "state.disk", False, f"low disk — {detail}", severity="warn")
    else:
        _note(checks, "state.disk", True, detail)


def _check_journals(checks: list[DoctorCheck], state_path: Path | None) -> None:
    """Replay every top-level ``*.jsonl`` journal — hash-chain verify plus
    torn-tail detection. Read-only: ``replay()`` never writes."""
    if state_path is None:
        _note(checks, "state.journals", True, "skipped (in-memory)")
        return
    if not state_path.is_dir():
        _note(checks, "state.journals", True, "no journals yet")
        return
    files = sorted(state_path.glob("*.jsonl"))
    if not files:
        _note(checks, "state.journals", True, "no journals yet")
        return
    parts: list[str] = []
    broken: list[str] = []
    for path in files:
        try:
            res = JobJournal(path).replay()
        except OSError as exc:
            broken.append(f"{path.name}: unreadable ({exc})")
            continue
        parts.append(f"{path.name}={len(res.payloads)}")
        if res.dropped:
            broken.append(
                f"{path.name}: chain broke at byte {res.truncated_at} "
                f"({res.dropped} line(s) dropped)"
            )
    detail = f"{len(files)} journal(s), entries: " + ", ".join(parts)
    if broken:
        _note(
            checks,
            "state.journals",
            False,
            detail + " — " + "; ".join(broken),
            severity="warn",
        )
    else:
        _note(checks, "state.journals", True, detail)


def _check_keys(checks: list[DoctorCheck], key_store: ApiKeyStore) -> None:
    records = key_store.list()
    now = time.time()
    enabled = [r for r in records if r.get("enabled")]
    revoked = [r for r in records if not r.get("enabled")]
    expired = [
        r
        for r in enabled
        if isinstance(r.get("expires_at"), (int, float)) and r["expires_at"] <= now
    ]
    usable = [r for r in enabled if r not in expired]
    env_key = bool(os.environ.get("FX1_API_KEY", ""))
    _note(
        checks,
        "keys.inventory",
        True,
        f"{len(records)} managed key(s): {len(enabled)} enabled, "
        f"{len(revoked)} revoked, {len(expired)} expired"
        + ("; env credential configured" if env_key else ""),
    )
    if env_key:
        _note(checks, "keys.auth", True, "env credential authenticates")
    elif not records:
        _note(
            checks,
            "keys.auth",
            True,
            "auth disabled — loopback clients only (dev mode)",
        )
    elif usable:
        _note(
            checks,
            "keys.auth",
            True,
            f"{len(usable)} usable managed credential(s)",
        )
    else:
        _note(
            checks,
            "keys.auth",
            False,
            "auth is on but no credential can authenticate "
            f"({len(revoked)} revoked, {len(expired)} expired)",
        )

    exhausted: list[str] = []
    saturated: list[str] = []
    for rec in usable:
        key_id = rec.get("id")
        max_req = rec.get("max_requests")
        max_tok = rec.get("max_tokens")
        if (isinstance(max_req, int) and rec.get("uses", 0) >= max_req) or (
            isinstance(max_tok, int) and rec.get("tokens_used", 0) >= max_tok
        ):
            exhausted.append(str(key_id))
            continue
        if key_id is None:
            continue
        window = key_store.window_state(str(key_id))
        if window is not None and window[1] <= 0:
            saturated.append(str(key_id))
    if not exhausted and not saturated:
        _note(checks, "keys.quota", True, "no exhausted budgets or saturated windows")
        return
    bits = []
    if exhausted:
        bits.append(f"quota-exhausted: {', '.join(exhausted[:8])}")
    if saturated:
        bits.append(f"rpm-saturated: {', '.join(saturated[:8])}")
    _note(checks, "keys.quota", False, "; ".join(bits), severity="warn")


def _check_capacity(
    checks: list[DoctorCheck], metrics: _OpsMetrics | None, registered_commands: int
) -> None:
    if metrics is None:
        _note(
            checks,
            "capacity",
            True,
            "in-process leg — no HTTP pool or drain latch",
        )
    else:
        snap = metrics.snapshot()
        detail = (
            f"inflight {snap.inflight}/{snap.max_inflight}, "
            f"watermark {snap.inflight_watermark}, uptime {snap.uptime_s:.0f}s, "
            f"rate-limited {snap.rate_limited_total}"
        )
        if snap.draining:
            _note(
                checks,
                "capacity",
                False,
                "drain latched — gated routes refuse new work; " + detail,
                severity="warn",
            )
        elif snap.max_inflight and snap.inflight >= snap.max_inflight:
            _note(
                checks,
                "capacity",
                False,
                "pool saturated — " + detail,
                severity="warn",
            )
        else:
            _note(checks, "capacity", True, detail)
    _note(
        checks,
        "harness.commands",
        registered_commands > 0,
        (
            f"{registered_commands} registered commands"
            if registered_commands
            else "empty command registry — harness cannot run work"
        ),
    )


def _check_version(checks: list[DoctorCheck]) -> None:
    _note(
        checks,
        "version.build",
        True,
        f"fx1 {__version__}, api {API_VERSION}",
    )


def build_doctor_report(
    *,
    mode: Literal["server", "in_process"],
    key_store: ApiKeyStore,
    state_path: Path | None = None,
    metrics: _OpsMetrics | None = None,
    registered_commands: int = 0,
    probe_backends: bool = True,
    timeout_s: float = _DOCTOR_PROBE_TIMEOUT_S,
) -> DoctorReport:
    """Run the deployment diagnosis and return the structured verdict.

    Read-only on every leg: journals are replayed (never appended),
    stores are only listed, and the BYOK probe is one cheap GET on the
    provider's ``/models`` route — never a paid completion.
    """
    checks: list[DoctorCheck] = []
    _check_backends(checks, probe_backends=probe_backends, timeout_s=timeout_s)
    _check_state_dir(checks, state_path)
    _check_journals(checks, state_path)
    _check_keys(checks, key_store)
    _check_capacity(checks, metrics, registered_commands)
    _check_version(checks)
    return DoctorReport(
        mode=mode,
        checked_at=time.time(),
        verdict=doctor_verdict(checks),
        checks=checks,
    )


def render_check_table(report: DoctorReport) -> str:
    """The human view for the CLI: one aligned line per check plus the
    verdict — mirrors the JSON's fields exactly."""
    lines = [
        f"fx1 harness doctor — verdict: {report.verdict} "
        f"(mode={report.mode}, {len(report.checks)} checks)"
    ]
    width = max((len(c.name) for c in report.checks), default=8)
    for check in report.checks:
        mark = "ok" if check.ok else ("FAIL" if check.severity == "error" else "warn")
        lines.append(f"  {check.name:<{width}}  {mark:<4}  {check.detail}")
    return "\n".join(lines)
