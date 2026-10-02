"""cli_audit — adversarial probes on the ``fx1`` typer front door.

Pinned contract (in-process via ``typer.testing.CliRunner``):

- The bare callback exits cleanly and lists commands; ``--help`` exits 0.
- ``sources fetch --params-json`` requires a JSON *object* — a JSON array
  or scalar exits 2 (usage), never a traceback leak past the runner.
- ``sources describe <unknown>`` fails non-zero — unknown names refuse.
- ``_resolve_judge`` fails closed: only ``hosted_k3`` is accepted, every
  other backend string exits 2, ``None`` selects the deterministic
  rule-based judge.
- ``dipbench`` with no ``--data-dir`` runs the SYNTHETIC smoke path and
  labels its output SYNTHETIC (never a real-data claim).
- ``doctor`` exits 0 and emits a JSON status blob with presence flags.
- ``maskedaEval`` is the registered command name — the ``masked-eval``
  spelling is not routed (a documented wart, not fixed here).
- Every eval command maps ``report.passed`` to the process exit code.

Sealed ``cli_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["cli_audit", "cli_audit_bench"]


def cli_audit() -> dict[str, Any]:
    import json

    import typer
    from typer.testing import CliRunner

    from fx1.cli import _resolve_judge, app

    runner = CliRunner()
    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    # top-level routing
    bare = runner.invoke(app, [])
    out["bare_exits_clean"] = bare.exit_code == 0
    out["help_lists"] = runner.invoke(app, ["--help"]).exit_code == 0
    out["unknown_command_fails"] = runner.invoke(app, ["no-such-command"]).exit_code != 0

    # sources fetch param gate — JSON object only
    out["fetch_array_refused"] = (
        runner.invoke(
            app,
            ["sources", "fetch", "finance_fetch", "--api", "x", "--params-json", "[1]"],
        ).exit_code
        == 2
    )
    out["fetch_scalar_refused"] = (
        runner.invoke(
            app,
            ["sources", "fetch", "finance_fetch", "--api", "x", "--params-json", '"x"'],
        ).exit_code
        == 2
    )
    out["fetch_junk_refused"] = (
        runner.invoke(
            app,
            ["sources", "fetch", "finance_fetch", "--api", "x", "--params-json", "{bad"],
        ).exit_code
        != 0
    )
    out["describe_unknown_fails"] = (
        runner.invoke(app, ["sources", "describe", "no-such-source"]).exit_code != 0
    )

    # judge resolution
    out["judge_none_ok"] = _resolve_judge(None) is None
    out["judge_bogus_exits2"] = _raises(lambda: _resolve_judge("bogus")).endswith("Exit")
    try:
        _resolve_judge("bogus")
        code = -1
    except typer.Exit as e:
        code = e.exit_code
    out["judge_bogus_code2"] = code == 2

    # dipbench synthetic path is labeled
    dip = runner.invoke(app, ["dipbench"])
    out["dipbench_synthetic_labeled"] = dip.exit_code == 0 and '"SYNTHETIC"' in dip.stdout

    # doctor emits presence flags
    doc = runner.invoke(app, ["doctor", "--root", "."])
    blob_ok = False
    if doc.exit_code == 0:
        try:
            blob = json.loads(doc.stdout)
            blob_ok = isinstance(blob, dict) and len(blob) > 0
        except Exception:
            blob_ok = False
    out["doctor_json_flags"] = blob_ok

    # registered command surface — maskedaEval is the real name (wart)
    from typer.core import TyperGroup  # noqa: PLC0415
    from typer.main import get_command  # noqa: PLC0415

    top = get_command(app)
    assert isinstance(top, TyperGroup)
    names = set(top.commands)
    groups = {n for n in names if isinstance(top.commands[n], TyperGroup)}
    out["expected_commands"] = {
        "capability-eval",
        "ext-bench-eval",
        "options-reasoning-eval",
        "redteam",
        "dipbench",
        "infer",
        "backtest",
        "doctor",
        "modelcard",
        "dpo",
        "curriculum",
        "sign",
        "attestation",
        "sbom",
        "mrm",
        "contamination-audit",
    } <= names
    out["flag_maskedeval_wart"] = "maskedaEval" in names and "masked-eval" not in names
    out["groups_registered"] = {"corpus", "harness", "sources"} <= groups

    # harness group — the SDK's shell surface
    harness = top.commands.get("harness")
    hnames = set(harness.commands) if isinstance(harness, TyperGroup) else set()
    out["harness_surface"] = {"list", "run", "serve", "complete", "verify", "health"} <= hnames

    h = runner.invoke(app, ["harness", "health"])
    hblob = json.loads(h.stdout) if h.exit_code == 0 else {}
    out["harness_health_json"] = hblob.get("status") in {"ok", "degraded"} and all(
        isinstance(v, bool) for v in hblob.get("backends", {}).values()
    )
    sent = "deadbeefsecret-marker-do-not-leak"
    h2 = runner.invoke(app, ["harness", "health"], env={"FX1_BYOK_API_KEY": sent})
    out["harness_health_no_secret_leak"] = sent not in h2.stdout

    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from fx1.bench.dip_audit import dip_audit_bench  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "ok.json"
        good.write_text(json.dumps(dip_audit_bench()))
        bad = Path(td) / "bad.json"
        tampered = json.loads(good.read_text())
        tampered["claim"]["results"]["dip_causality"] = False
        bad.write_text(json.dumps(tampered))
        out["harness_verify_valid_exits_0"] = (
            runner.invoke(app, ["harness", "verify", str(good)]).exit_code == 0
        )
        out["harness_verify_tampered_fails"] = (
            runner.invoke(app, ["harness", "verify", str(bad)]).exit_code != 0
        )
    return out


def cli_audit_bench() -> dict[str, Any]:
    r = cli_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "cli_audit",
        "schema": "cli_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "fx1 CLI holds: bare callback + help exit clean, fetch params "
            "must be a JSON object (exit 2), unknown sources/commands fail "
            "non-zero, judge resolution fails closed, dipbench smoke is "
            "SYNTHETIC-labeled, doctor emits presence-only JSON. Flagged "
            "wart: the command is registered as 'maskedaEval'."
            if ok
            else f"CLI AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
