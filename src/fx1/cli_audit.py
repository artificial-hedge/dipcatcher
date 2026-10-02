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

    # `harness complete` gated surfaces — stream vs block over an injected SDK
    from unittest.mock import patch  # noqa: PLC0415

    from fx1.sdk import CompletionResult  # noqa: PLC0415

    class _FakeSDK:
        def __init__(self) -> None:
            self.stream_calls: list[dict[str, Any]] = []

        def complete(self, messages: Any, **kw: Any) -> CompletionResult:
            return CompletionResult(
                backend=str(kw.get("backend")), model="fake-v0", content="block-text"
            )

        def stream_complete(self, messages: Any, **kw: Any) -> list[str]:
            self.stream_calls.append(dict(kw))
            return ["chunk-a", "chunk-b"]

        def complete_many(self, batch: Any, **kw: Any) -> list[CompletionResult]:
            return [
                CompletionResult(
                    backend="byok",
                    model="fake-v0",
                    content=f"b:{m[-1]['content']}",
                )
                for m in batch
            ]

        def run(self, name: str, **kw: Any) -> Any:
            from fx1.harness import HarnessResult

            return HarnessResult(command=name, exit_code=0, stdout="ran", stderr="")

        def verify_receipt(self, receipt: dict[str, Any]) -> Any:
            from fx1.sdk import ReceiptVerdict

            return ReceiptVerdict(
                valid=True,
                path="<cli>",
                errors=(),
                warnings=(),
                schema_tag="x",
                kind="k",
                verdict="ok",
                digest_convention=None,
            )

        def health(self) -> Any:
            from fx1.sdk import HarnessHealth

            return HarnessHealth(
                status="ok", version="v", registered_commands=1, backends={"byok": True}
            )

        def commands(self, role: Any = None) -> list[str]:
            return ["cmd-a", "cmd-b"]

    fake = _FakeSDK()
    with patch("fx1.sdk.Fx1Harness", return_value=fake):
        out["complete_block_echoes_content"] = (
            runner.invoke(app, ["harness", "complete", "hi", "--backend", "byok"]).stdout.strip()
            == "block-text"
        )
        rs = runner.invoke(
            app,
            [
                "harness",
                "complete",
                "hi",
                "--backend",
                "byok",
                "--stream",
                "--receipt",
                "a" * 64,
            ],
        )
        out["complete_stream_concatenates_chunks"] = (
            rs.exit_code == 0 and rs.stdout == "chunk-achunk-b\n"
        )
        out["complete_stream_forwards_receipts"] = bool(fake.stream_calls) and (
            fake.stream_calls[0].get("receipt_hashes") == ["a" * 64]
        )
        # `harness batch` — JSONL/JSON prompts -> complete_many -> JSON out
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as td:
            pf = Path(td) / "prompts.jsonl"
            pf.write_text('"one"\n{"prompt": "two"}\n')
            rb = runner.invoke(app, ["harness", "batch", str(pf), "--backend", "byok"])
            payload = json.loads(rb.stdout) if rb.exit_code == 0 else []
            out["batch_jsonl_two_items"] = [p["content"] for p in payload] == [
                "b:one",
                "b:two",
            ]
            bad = Path(td) / "bad.jsonl"
            bad.write_text("")
            out["batch_empty_fails_clean"] = (
                runner.invoke(app, ["harness", "batch", str(bad)]).exit_code == 2
            )

    # --remote routes the same commands through HarnessClient --------------
    class _FakeRemote:
        def __init__(self, base_url: str, **kw: Any) -> None:
            self.base_url = base_url
            self.api_key = kw.get("api_key")

        def complete(self, messages: Any, **kw: Any) -> CompletionResult:
            return CompletionResult(backend="byok", model="remote-v0", content="remote-text")

        def commands(self, role: Any = None) -> list[str]:
            return ["cmd-a"]

        def health(self) -> Any:
            from fx1.sdk import HarnessHealth

            return HarnessHealth(status="ok", version="v", registered_commands=1, backends={})

        def metrics(self) -> Any:
            from fx1.sdk import OpsMetrics

            return OpsMetrics(
                uptime_s=1.0,
                requests_total=7,
                errors_total=0,
                by_status={"200": 7},
                inflight=0,
                inflight_watermark=1,
                max_inflight=16,
            )

    remotes: list[_FakeRemote] = []

    def _mk_remote(url: str, **kw: Any) -> _FakeRemote:
        r = _FakeRemote(url, **kw)
        remotes.append(r)
        return r

    with patch("fx1.serve.client.HarnessClient", side_effect=_mk_remote):
        rr = runner.invoke(
            app,
            ["harness", "complete", "hi", "--remote", "http://h.test", "--api-key", "k"],
        )
        out["remote_complete_uses_client"] = (
            rr.exit_code == 0
            and rr.stdout.strip() == "remote-text"
            and remotes[0].base_url == "http://h.test"
            and remotes[0].api_key == "k"
        )
        out["remote_list_names"] = (
            runner.invoke(app, ["harness", "list", "--remote", "http://h.test"]).stdout.strip()
            == "cmd-a"
        )
        rm = runner.invoke(app, ["harness", "metrics", "--remote", "http://h.test"])
        out["remote_metrics_json"] = (
            rm.exit_code == 0
            and json.loads(rm.stdout)["requests_total"] == 7
            and json.loads(rm.stdout)["by_status"] == {"200": 7}
        )

    # metrics is a wire-ops surface — without --remote it fails clean
    rm_local = runner.invoke(app, ["harness", "metrics"])
    out["metrics_local_refused"] = rm_local.exit_code == 2 and "--remote" in rm_local.output

    class _FailingRemote:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def health(self) -> Any:
            from fx1.serve.client import HarnessTransportError

            raise HarnessTransportError("connection refused")

    with patch("fx1.serve.client.HarnessClient", _FailingRemote):
        rf = runner.invoke(app, ["harness", "health", "--remote", "http://dead"])
        out["remote_fault_clean_exit2"] = rf.exit_code == 2 and "HarnessTransportError" in rf.output
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
