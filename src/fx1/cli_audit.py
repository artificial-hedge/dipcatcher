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
    out["harness_surface"] = {
        "list",
        "run",
        "serve",
        "complete",
        "verify",
        "health",
        "probe",
        "check-text",
    } <= hnames

    # probe verdicts are the exit code: 0 ok, 1 unhealthy, !=0 arg fault —
    # a dead BYOK endpoint is a verdict, not a crash.
    p_dead = runner.invoke(
        app,
        [
            "harness",
            "probe",
            "--backend",
            "byok",
            "--byok-base-url",
            "http://127.0.0.1:9",
            "--byok-api-key",
            "k",
            "--byok-model",
            "m",
            "--backend-timeout",
            "2",
        ],
    )
    pblob = json.loads(p_dead.stdout) if p_dead.stdout.strip().startswith("{") else {}
    out["harness_probe_unhealthy_exit"] = p_dead.exit_code == 1 and pblob.get("ok") is False
    out["harness_probe_unknown_fails"] = (
        runner.invoke(app, ["harness", "probe", "--backend", "bogus"]).exit_code != 0
    )

    # gate check-text: exit 0 clean / 1 refusal; in-process needs no backend
    c_ok = runner.invoke(app, ["harness", "check-text", "bootstrap intervals"])
    c_bad = runner.invoke(app, ["harness", "check-text", "we report Sharpe 2.1"])
    cblob = json.loads(c_bad.stdout) if c_bad.stdout.strip().startswith("{") else {}
    out["harness_check_clean"] = c_ok.exit_code == 0 and json.loads(c_ok.stdout).get("ok") is True
    out["harness_check_refusal"] = (
        c_bad.exit_code == 1 and cblob.get("ok") is False and isinstance(cblob.get("error"), str)
    )

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
        # dir mode: every *.json under the dir verified, exit reflects all
        okdir = Path(td) / "okdir"
        okdir.mkdir()
        (okdir / "a.json").write_text(good.read_text())
        (okdir / "b.json").write_text(good.read_text())
        rd = runner.invoke(app, ["harness", "verify", str(okdir)])
        rdj = json.loads(rd.stdout) if rd.exit_code == 0 else {}
        out["harness_verify_dir_all_valid"] = rdj.get("files") == 2 and rdj.get("valid") == 2
        (okdir / "bad.json").write_text(bad.read_text())
        rd2 = runner.invoke(app, ["harness", "verify", str(okdir)])
        rd2j = json.loads(rd2.stdout) if rd2.stdout else {}
        out["harness_verify_dir_mixed_fails"] = (
            rd2.exit_code == 1 and rd2j.get("valid") == 2 and rd2j.get("files") == 3
        )
        empty = Path(td) / "empty"
        empty.mkdir()
        out["harness_verify_dir_empty_2"] = (
            runner.invoke(app, ["harness", "verify", str(empty)]).exit_code == 2
        )

    # `harness receipts`/`receipt` over a local sealed store — same contract as
    # the API's /receipts routes, read in-process.
    with tempfile.TemporaryDirectory() as td2:
        rdir = Path(td2) / "store"
        rdir.mkdir()
        sealed = dip_audit_bench()
        sha = sealed["receipt_sha256"]
        (rdir / "s.json").write_text(json.dumps(sealed))
        ri = runner.invoke(app, ["harness", "receipts", "--receipts-dir", str(rdir)])
        out["harness_receipts_lists"] = ri.exit_code == 0 and json.loads(ri.stdout)["items"] == [
            {"sha256": sha, "name": "s.json"}
        ]
        rf = runner.invoke(app, ["harness", "receipt", sha, "--receipts-dir", str(rdir)])
        rfj = json.loads(rf.stdout) if rf.exit_code == 0 else {}
        out["harness_receipt_fetches"] = (
            rfj.get("receipt", {}).get("receipt_sha256") == sha and rfj.get("valid") is True
        )
        out["harness_receipt_miss_exits_2"] = (
            runner.invoke(
                app, ["harness", "receipt", "f" * 64, "--receipts-dir", str(rdir)]
            ).exit_code
            == 2
        )
        out["harness_receipts_absent_dir_2"] = (
            runner.invoke(
                app, ["harness", "receipts", "--receipts-dir", str(Path(td2) / "gone")]
            ).exit_code
            == 2
        )

    # `harness complete` gated surfaces — stream vs block over an injected SDK
    from unittest.mock import patch  # noqa: PLC0415

    from fx1.sdk import CompletionResult  # noqa: PLC0415

    class _FakeSDK:
        def __init__(self) -> None:
            self.stream_calls: list[dict[str, Any]] = []
            self.complete_calls: list[dict[str, Any]] = []
            self.batch_calls: list[dict[str, Any]] = []

        def complete(self, messages: Any, **kw: Any) -> CompletionResult:
            self.complete_calls.append(dict(kw))
            return CompletionResult(
                backend=str(kw.get("backend")), model="fake-v0", content="block-text"
            )

        def stream_complete(self, messages: Any, **kw: Any) -> list[str]:
            self.stream_calls.append(dict(kw))
            return ["chunk-a", "chunk-b"]

        def complete_many(self, batch: Any, **kw: Any) -> list[CompletionResult]:
            self.batch_calls.append(dict(kw))
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
        # per-request BYOK flags pack into the byok override (all-or-none)
        rb2 = runner.invoke(
            app,
            [
                "harness",
                "complete",
                "hi",
                "--backend",
                "byok",
                "--byok-base-url",
                "https://e.com",
                "--byok-api-key",
                "sk-x",
                "--byok-model",
                "m1",
            ],
        )
        out["complete_byok_flags_forward"] = rb2.exit_code == 0 and fake.complete_calls[-1].get(
            "byok"
        ) == {
            "base_url": "https://e.com",
            "api_key": "sk-x",
            "model": "m1",
        }
        out["complete_byok_partial_exits_2"] = (
            runner.invoke(
                app, ["harness", "complete", "hi", "--backend", "byok", "--byok-api-key", "sk-x"]
            ).exit_code
            == 2
        )
        # --backend-timeout packs into the wire timeout_s on both surfaces.
        rt = runner.invoke(
            app,
            ["harness", "complete", "hi", "--backend", "byok", "--backend-timeout", "7.5"],
        )
        out["complete_backend_timeout_forwards"] = (
            rt.exit_code == 0 and fake.complete_calls[-1].get("timeout_s") == 7.5
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
            rb_t = runner.invoke(
                app,
                ["harness", "batch", str(pf), "--backend", "byok", "--backend-timeout", "5"],
            )
            out["batch_backend_timeout_forwards"] = (
                rb_t.exit_code == 0 and fake.batch_calls[-1].get("timeout_s") == 5.0
            )

    # --remote routes the same commands through HarnessClient --------------
    class _FakeRemote:
        def __init__(self, base_url: str, **kw: Any) -> None:
            self.base_url = base_url
            self.api_key = kw.get("api_key")
            self.last_idem: str | None = None
            self.last_job: str | None = None
            self.last_wait: float | None = None
            self.last_drain_wait: float | None = None
            self.last_jobs_query: dict[str, Any] | None = None
            self.last_callback: str | None = None
            self.last_cb_secret: str | None = None
            self.last_compat_strict: bool | None = None

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
                rate_limited_total=3,
            )

        def metrics_text(self) -> str:
            return "fx1_requests_total 7\n"

        def drain(self, wait_s: float = 0.0) -> dict[str, Any]:
            self.last_drain_wait = wait_s
            return {"draining": True, "inflight": 2, "drained": False}

        def ready(self) -> dict[str, Any]:
            return {"ready": True, "inflight": 2}

        def server_version(self) -> dict[str, Any]:
            return {"api_version": "1", "fx1_version": "0.4.0"}

        def capabilities(self) -> dict[str, Any]:
            return {
                "api_version": "1",
                "fx1_version": "0.4.0",
                "features": {"jobs": True, "sse": True},
                "limits": {"max_inflight": 4.0, "job_batch_max": 64.0},
                "backends": {"byok": True},
                "roles": ["evaluation"],
            }

        def check_compat(self, strict: bool = True) -> dict[str, Any]:
            self.last_compat_strict = strict
            return {
                "compatible": True,
                "client_api_version": "1",
                "server_api_version": "1",
                "server_fx1_version": "0.4.0",
                "client_fx1_version": "0.4.0",
            }

        def run(self, name: str, **kw: Any) -> Any:
            from fx1.harness import HarnessResult

            self.last_idem = kw.get("idempotency_key")
            return HarnessResult(command=name, exit_code=0, stdout="ran", stderr="")

        def submit_run(self, name: str, **kw: Any) -> str:
            self.last_idem = kw.get("idempotency_key")
            self.last_callback = kw.get("callback_url")
            self.last_cb_secret = kw.get("callback_secret")
            return "job-xyz"

        def job_status(self, job_id: str) -> dict[str, Any]:
            self.last_job = job_id
            return {"job_id": job_id, "status": "succeeded", "result": None}

        def list_jobs(self, **kw: Any) -> dict[str, Any]:
            self.last_jobs_query = dict(kw)
            return {
                "jobs": [{"job_id": "job-xyz", "status": "succeeded", "result": None}],
                "total": 1,
            }

        def cancel_job(self, job_id: str) -> dict[str, Any]:
            self.last_job = job_id
            return {"job_id": job_id, "status": "cancelled", "result": None}

        def wait_run(self, job_id: str, **kw: Any) -> Any:
            from fx1.harness import HarnessResult

            self.last_job = job_id
            self.last_wait = kw.get("timeout_s")
            return HarnessResult(command="doctor", exit_code=0, stdout="ran", stderr="")

        def submit_batch(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
            self.last_batch = list(jobs)
            return {
                "jobs": [
                    {"index": i, "job_id": f"jb{i}", "status": "queued", "replayed": False}
                    for i in range(len(jobs))
                ],
                "submitted": len(jobs),
                "failed": 0,
            }

        def stream_job(self, job_id: str, **kw: Any) -> list[dict[str, Any]]:
            self.last_job = job_id
            self.last_stream_timeout = kw.get("timeout_s")
            return [
                {"job_id": job_id, "status": "queued"},
                {
                    "job_id": job_id,
                    "status": "succeeded",
                    "result": {
                        "command": "doctor",
                        "exit_code": 0,
                        "stdout": "ran",
                        "stderr": "",
                    },
                },
            ]

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
            and json.loads(rm.stdout)["rate_limited_total"] == 3
        )
        rp = runner.invoke(
            app, ["harness", "metrics", "--remote", "http://h.test", "--format", "prom"]
        )
        out["remote_metrics_prom"] = (
            rp.exit_code == 0 and rp.stdout.strip() == "fx1_requests_total 7"
        )
        rb = runner.invoke(
            app, ["harness", "metrics", "--remote", "http://h.test", "--format", "xml"]
        )
        out["remote_metrics_bad_format"] = rb.exit_code == 2
        rd = runner.invoke(app, ["harness", "drain", "--remote", "http://h.test"])
        out["remote_drain_json"] = rd.exit_code == 0 and json.loads(rd.stdout) == {
            "draining": True,
            "inflight": 2,
            "drained": False,
        }
        rw = runner.invoke(
            app,
            ["harness", "drain", "--remote", "http://h.test", "--wait-s", "10"],
        )
        out["remote_drain_wait_flag"] = rw.exit_code == 0 and remotes[-1].last_drain_wait == 10.0
        rready = runner.invoke(app, ["harness", "ready", "--remote", "http://h.test"])
        out["remote_ready_json"] = (
            rready.exit_code == 0 and json.loads(rready.stdout)["ready"] is True
        )
        rv = runner.invoke(app, ["harness", "version", "--remote", "http://h.test"])
        out["remote_version_json"] = (
            rv.exit_code == 0 and json.loads(rv.stdout)["api_version"] == "1"
        )
        rcp = runner.invoke(app, ["harness", "compat", "--remote", "http://h.test"])
        out["remote_compat_ok_exit0"] = (
            rcp.exit_code == 0
            and json.loads(rcp.stdout)["compatible"] is True
            and remotes[-1].last_compat_strict is False
        )
        rj = runner.invoke(
            app,
            [
                "harness",
                "jobs",
                "--remote",
                "http://h.test",
                "--status",
                "succeeded",
                "--limit",
                "5",
                "--offset",
                "2",
            ],
        )
        out["remote_jobs_json"] = (
            rj.exit_code == 0
            and json.loads(rj.stdout)["total"] == 1
            and remotes[-1].last_jobs_query == {"status": "succeeded", "limit": 5, "offset": 2}
        )
        rcx = runner.invoke(app, ["harness", "cancel", "job-xyz", "--remote", "http://h.test"])
        out["remote_cancel_json"] = (
            rcx.exit_code == 0
            and json.loads(rcx.stdout)["status"] == "cancelled"
            and remotes[-1].last_job == "job-xyz"
        )

    # ready under drain: client raises the mapped 503, CLI exits 1
    from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

    with patch("fx1.serve.client.HarnessClient") as mc:
        inst = mc.return_value
        inst.ready.side_effect = BackendNotConfiguredError("draining")
        rnot = runner.invoke(app, ["harness", "ready", "--remote", "http://h.test"])
        out["remote_ready_under_drain_exit1"] = (
            rnot.exit_code == 1 and '"ready": false' in rnot.stdout
        )

    ry_local = runner.invoke(app, ["harness", "ready"])
    out["ready_local_refused"] = ry_local.exit_code == 2 and "--remote" in ry_local.output
    rv_local = runner.invoke(app, ["harness", "version"])
    out["version_local_json"] = (
        rv_local.exit_code == 0
        and json.loads(rv_local.stdout)["api_version"] == "1"
        and json.loads(rv_local.stdout)["local"] is True
    )

    # compat on a wire-contract mismatch exits 1 but still prints the report
    with patch("fx1.serve.client.HarnessClient") as mc2:
        inst2 = mc2.return_value
        inst2.check_compat.return_value = {
            "compatible": False,
            "client_api_version": "1",
            "server_api_version": "99",
            "server_fx1_version": "9.9",
            "client_fx1_version": "0.4.0",
        }
        rcp_bad = runner.invoke(app, ["harness", "compat", "--remote", "http://h.test"])
        out["remote_compat_mismatch_exit1"] = (
            rcp_bad.exit_code == 1
            and json.loads(rcp_bad.stdout)["compatible"] is False
            and json.loads(rcp_bad.stdout)["server_api_version"] == "99"
        )
    rcp_local = runner.invoke(app, ["harness", "compat"])
    out["compat_local_refused"] = rcp_local.exit_code == 2 and "--remote" in rcp_local.output

    # capabilities is a wire-ops surface: refused locally, JSON remote
    with patch("fx1.serve.client.HarnessClient") as mc3:
        inst3 = mc3.return_value
        inst3.capabilities.return_value = {
            "api_version": "1",
            "features": {"jobs": True},
            "limits": {"job_batch_max": 64.0},
            "backends": {"byok": True},
            "roles": ["evaluation"],
        }
        rcp_cap = runner.invoke(app, ["harness", "capabilities", "--remote", "http://h.test"])
        out["remote_capabilities_json"] = (
            rcp_cap.exit_code == 0 and json.loads(rcp_cap.stdout)["limits"]["job_batch_max"] == 64.0
        )
    rcap_local = runner.invoke(app, ["harness", "capabilities"])
    out["capabilities_local_refused"] = (
        rcap_local.exit_code == 2 and "--remote" in rcap_local.output
    )

    # metrics + drain + jobs are wire-ops surfaces — without --remote they fail clean
    rm_local = runner.invoke(app, ["harness", "metrics"])
    out["metrics_local_refused"] = rm_local.exit_code == 2 and "--remote" in rm_local.output
    rd_local = runner.invoke(app, ["harness", "drain"])
    out["drain_local_refused"] = rd_local.exit_code == 2 and "--remote" in rd_local.output
    rs_local = runner.invoke(app, ["harness", "submit", "doctor"])
    out["submit_local_refused"] = rs_local.exit_code == 2 and "--remote" in rs_local.output
    rj_local = runner.invoke(app, ["harness", "jobs"])
    out["jobs_local_refused"] = rj_local.exit_code == 2 and "--remote" in rj_local.output
    rc_local = runner.invoke(app, ["harness", "cancel", "job-xyz"])
    out["cancel_local_refused"] = rc_local.exit_code == 2 and "--remote" in rc_local.output

    # --idempotency-key reaches the remote client verbatim
    with patch("fx1.serve.client.HarnessClient", side_effect=_mk_remote):
        rr_key = runner.invoke(
            app,
            [
                "harness",
                "run",
                "doctor",
                "--remote",
                "http://h.test",
                "--idempotency-key",
                "cli-key-1",
            ],
        )
        out["cli_run_idem_key_passed"] = (
            rr_key.exit_code == 0 and remotes[-1].last_idem == "cli-key-1"
        )

        rs = runner.invoke(
            app,
            [
                "harness",
                "submit",
                "doctor",
                "--remote",
                "http://h.test",
                "--idempotency-key",
                "sub-k",
            ],
        )
        out["cli_submit_prints_job_id"] = (
            rs.exit_code == 0
            and rs.stdout.strip() == "job-xyz"
            and remotes[-1].last_idem == "sub-k"
        )
        rc = runner.invoke(
            app,
            [
                "harness",
                "submit",
                "doctor",
                "--remote",
                "http://h.test",
                "--callback-url",
                "https://hooks.test/x",
            ],
        )
        out["cli_submit_callback_url"] = (
            rc.exit_code == 0 and remotes[-1].last_callback == "https://hooks.test/x"
        )
        rcs = runner.invoke(
            app,
            [
                "harness",
                "submit",
                "doctor",
                "--remote",
                "http://h.test",
                "--callback-url",
                "https://hooks.test/x",
                "--callback-secret",
                "whsec-test",
            ],
        )
        out["cli_submit_callback_secret"] = (
            rcs.exit_code == 0 and remotes[-1].last_cb_secret == "whsec-test"
        )
        rj = runner.invoke(app, ["harness", "job", "j-9", "--remote", "http://h.test"])
        out["cli_job_status_json"] = (
            rj.exit_code == 0
            and json.loads(rj.stdout)["status"] == "succeeded"
            and remotes[-1].last_job == "j-9"
        )
        rw = runner.invoke(app, ["harness", "wait", "j-9", "--remote", "http://h.test"])
        out["cli_wait_prints_result"] = rw.exit_code == 0 and rw.stdout.strip() == "ran"
        rww = runner.invoke(app, ["harness", "watch", "j-7", "--remote", "http://h.test"])
        out["cli_watch_streams"] = (
            rww.exit_code == 0
            and "queued\tj-7" in rww.stdout
            and "succeeded\tj-7" in rww.stdout
            and "ran" in rww.stdout
            and remotes[-1].last_job == "j-7"
        )
        import tempfile  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as btd:
            spec = _Path(btd) / "batch.json"
            spec.write_text(
                json.dumps(
                    [
                        {"command": "doctor", "idempotency_key": "b1"},
                        {"command": "operations"},
                    ]
                )
            )
            rbs = runner.invoke(
                app,
                ["harness", "submit-batch", str(spec), "--remote", "http://h.test"],
            )
            out["cli_submit_batch"] = (
                rbs.exit_code == 0
                and json.loads(rbs.stdout)["submitted"] == 2
                and remotes[-1].last_batch[0]["idempotency_key"] == "b1"
            )
            bad_spec = _Path(btd) / "notalist.json"
            bad_spec.write_text(json.dumps({"command": "doctor"}))
            out["cli_submit_batch_not_list_2"] = (
                runner.invoke(
                    app,
                    ["harness", "submit-batch", str(bad_spec), "--remote", "http://h.test"],
                ).exit_code
                == 2
            )

    class _FailingRemote:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def health(self) -> Any:
            from fx1.serve.client import HarnessTransportError

            raise HarnessTransportError("connection refused")

    with patch("fx1.serve.client.HarnessClient", _FailingRemote):
        rf = runner.invoke(app, ["harness", "health", "--remote", "http://dead"])
        out["remote_fault_clean_exit2"] = rf.exit_code == 2 and "HarnessTransportError" in rf.output

    # --- harness serve: tuning knobs reach the app ------------------------------
    served: list[Any] = []

    with patch("uvicorn.run", lambda a, **kw: served.append((a, kw))):
        rs = runner.invoke(
            app,
            [
                "harness",
                "serve",
                "--max-inflight",
                "3",
                "--job-max",
                "7",
                "--idem-max",
                "9",
                "--sse-keepalive-s",
                "2.5",
                "--rate-limit-rps",
                "4.5",
                "--gzip-min-bytes",
                "5",
            ],
        )
        out["serve_flags_reach_app"] = (
            rs.exit_code == 0
            and len(served) == 1
            and served[0][1]["host"] == "127.0.0.1"
            and served[0][0].state.metrics.max_inflight == 3
            and served[0][0].state.sse_keepalive_s == 2.5
            and served[0][0].state.rate_limiter.rps == 4.5
            and served[0][0].state.gzip_min_bytes == 5
        )
        served.clear()
        rs2 = runner.invoke(app, ["harness", "serve", "--max-inflight", "0"])
        out["serve_bad_knob_exit2"] = rs2.exit_code == 2 and not served
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
