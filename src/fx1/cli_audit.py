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


def cli_audit() -> dict[str, Any]:  # noqa: C901 — probe accumulator
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
        "completions",
        "completion",
        "eval",
        "evals",
        "eval-status",
        "eval-cancel",
        "eval-wait",
        "eval-diff",
        "ft-create",
        "ft-jobs",
        "ft-status",
        "ft-events",
        "ft-wait",
        "ft-cancel",
        "ft-checkpoints",
        "files",
        "file-upload",
        "file-content",
        "file-delete",
        "batch-submit",
        "batches",
        "batch-status",
        "batch-cancel",
        "batch-output",
        "batch-run",
        "models",
        "model",
        "model-delete",
        "respond",
        "embed",
        "moderate",
        "chat-get",
        "chat-delete",
        "chat-list",
        "chat-messages",
        "response-get",
        "response-delete",
        "response-input-items",
        "score",
        "commands",
    } <= hnames

    # the files/batches family is wire-only — no --remote is exit 2 on
    # every command, never a traceback.
    out["harness_files_family_needs_remote"] = all(
        runner.invoke(app, ["harness", name, *args]).exit_code == 2
        for name, args in (
            ("files", []),
            ("file-upload", ["x.jsonl"]),
            ("file-content", ["file-x"]),
            ("file-delete", ["file-x"]),
            ("batch-submit", ["x.jsonl"]),
            ("batches", []),
            ("batch-status", ["batch_x"]),
            ("batch-wait", ["batch_x"]),
            ("batch-cancel", ["batch_x"]),
            ("batch-output", ["batch_x"]),
        )
    )
    # ft-create: callback_secret without callback_url refuses before any
    # network/backend work — the same guard as the wire's 422.
    import tempfile as _tmpf  # noqa: PLC0415
    from pathlib import Path as _P  # noqa: PLC0415

    with _tmpf.TemporaryDirectory() as _td:
        _cor = _P(_td) / "c.jsonl"
        _cor.write_text(
            '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
        )
        out["harness_ft_create_guard"] = (
            runner.invoke(
                app,
                ["harness", "ft-create", str(_cor), "--callback-secret", "x"],
            ).exit_code
            == 2
        )
        # a real corpus on the in-process SDK path exits 0 with the
        # terminal job record (synchronous stub runner).
        _ok = runner.invoke(app, ["harness", "ft-create", str(_cor), "--epochs", "1"])
        out["harness_ft_create_inprocess"] = _ok.exit_code == 0 and json.loads(_ok.stdout).get(
            "status"
        ) in {"succeeded", "failed"}

        # batch-run is the in-process twin — no --remote needed.
        _in = _P(_td) / "b.jsonl"
        _in.write_text(
            '{"custom_id":"r1","method":"POST","url":"/v1/chat/completions",'
            '"body":{"model":"local_fx1","messages":[{"role":"user","content":"q"}]}}\n'
        )
        out["batch_run_missing_file_2"] = (
            runner.invoke(app, ["harness", "batch-run", str(_P(_td) / "nope.jsonl")]).exit_code == 2
        )
        _bad = _P(_td) / "bad.jsonl"
        _bad.write_text("not json\n")
        out["batch_run_bad_jsonl_2"] = (
            runner.invoke(app, ["harness", "batch-run", str(_bad)]).exit_code == 2
        )
        _outp = _P(_td) / "out.jsonl"
        _rb = runner.invoke(app, ["harness", "batch-run", str(_in), "--out", str(_outp)])
        _blob = json.loads(_rb.stdout) if _rb.exit_code == 0 else {}
        out["batch_run_inprocess"] = (
            _rb.exit_code == 0
            and _blob.get("status") == "completed"
            and _outp.exists()
            and _outp.read_text().startswith("{")
        )
        # callback_secret alone → 2 (the wire's 422 twin)
        out["batch_run_guard"] = (
            runner.invoke(
                app, ["harness", "batch-run", str(_in), "--callback-secret", "x"]
            ).exit_code
            == 2
        )

        # model inventory: the list envelope names fx1 in-process too.
        _ml = runner.invoke(app, ["harness", "models"])
        out["harness_models_inprocess"] = _ml.exit_code == 0 and any(
            m.get("id") == "fx1" for m in json.loads(_ml.stdout).get("data", [])
        )
        out["harness_model_unknown_2"] = (
            runner.invoke(app, ["harness", "model", "nope-model"]).exit_code == 2
        )

        # respond/embed/moderate: in-process legs — bad JSON flags are arg
        # faults, a dead link is a clean 2 (never a traceback), and the
        # moderation gate needs no backend at all.
        out["harness_respond_bad_meta_2"] = (
            runner.invoke(app, ["harness", "respond", "hi", "--metadata", "[1]"]).exit_code == 2
        )
        out["harness_respond_bad_fmt_2"] = (
            runner.invoke(app, ["harness", "respond", "hi", "--format", "{bad"]).exit_code == 2
        )
        _rc = runner.invoke(app, ["harness", "respond", "hi", "--backend", "no-such"])
        out["harness_respond_deadlink_2"] = _rc.exit_code == 2 and _rc.stderr.startswith("error:")
        _ec = runner.invoke(app, ["harness", "embed", "hi", "--backend", "no-such"])
        out["harness_embed_deadlink_2"] = _ec.exit_code == 2 and _ec.stderr.startswith("error:")
        _mc = runner.invoke(app, ["harness", "moderate", "hello"])
        out["harness_moderate_inprocess"] = _mc.exit_code == 0 and json.loads(_mc.stdout).get(
            "id", ""
        ).startswith("modr-")
        _mf = runner.invoke(app, ["harness", "moderate", "our live trading sharpe is 9"])
        out["harness_moderate_flags"] = (
            _mf.exit_code == 0 and json.loads(_mf.stdout)["results"][0]["flagged"] is True
        )
        # stored-object retrieval: a missing id is a clean 2 in-process
        # (the wire's 404), never a fabricated envelope.
        out["harness_stored_missing_2"] = all(
            runner.invoke(app, ["harness", name, "no-such-id"]).exit_code == 2
            for name in ("chat-get", "chat-delete", "response-get", "response-delete")
        )
        # the stored-request subresources inherit the same contract —
        # missing id is a clean 2 in-process too.
        out["harness_items_missing_2"] = all(
            runner.invoke(app, ["harness", name, "no-such-id"]).exit_code == 2
            for name in ("chat-messages", "response-input-items")
        )
        # chat-list in-process: empty store lists [], malformed --metadata
        # is a clean 2.
        _cl_out = json.loads(runner.invoke(app, ["harness", "chat-list"]).stdout)
        out["harness_chat_list_inprocess"] = (
            _cl_out["object"] == "list"
            and _cl_out["data"] == []
            and _cl_out["has_more"] is False
            and runner.invoke(app, ["harness", "chat-list", "--metadata", "nokey"]).exit_code == 2
        )

        # score preflight: the reward contract runs in-process with no model
        # spend — banned-token text carries violations + the -10 total.
        _sc = json.loads(runner.invoke(app, ["harness", "score", "our live sharpe is 9"]).stdout)[0]
        out["harness_score_inprocess"] = bool(_sc["violations"]) and _sc["total"] < 0
        # commands registry: the bogus-role fault is exit 2 (wire's 422 twin).
        _cl = runner.invoke(app, ["harness", "commands", "--role", "evaluation"])
        out["harness_commands_role"] = _cl.exit_code == 0 and all(
            isinstance(n, str) for n in json.loads(_cl.stdout)["commands"]
        )
        out["harness_commands_bad_role_2"] = (
            runner.invoke(app, ["harness", "commands", "--role", "bogus"]).exit_code == 2
        )

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

    # completion log: in-process SDK records its own calls; the log starts
    # empty so `completions` prints a zero window and `completion` exits 2.
    c_empty = runner.invoke(app, ["harness", "completions"])
    out["harness_completions_empty"] = (
        c_empty.exit_code == 0 and json.loads(c_empty.stdout).get("count") == 0
    )
    out["harness_completion_missing"] = (
        runner.invoke(app, ["harness", "completion", "0" * 32]).exit_code == 2
        and runner.invoke(app, ["harness", "completion", "0" * 32, "--receipt"]).exit_code == 2
    )

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

        def openai_response_stream(
            self, request: Any, **kw: Any
        ) -> tuple[list[tuple[str, dict[str, Any]]], None]:
            self.stream_calls.append({"responses": True, **dict(kw)})
            return (
                [
                    ("response.output_text.delta", {"delta": "lo"}),
                    ("response.output_text.delta", {"delta": "cal"}),
                    ("response.completed", {"response": {}}),
                ],
                None,
            )

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

        def verify_receipts(self, receipts: list[dict[str, Any]]) -> Any:
            from fx1.sdk import ReceiptVerdict

            return tuple(
                ReceiptVerdict(
                    valid=i == 0,
                    path="<cli>",
                    errors=() if i == 0 else ("bad",),
                    warnings=(),
                    schema_tag="x",
                    kind="k",
                    verdict="ok" if i == 0 else "invalid",
                    digest_convention=None,
                )
                for i, _r in enumerate(receipts)
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
        # respond --stream in-process: SDK (event, payload) pairs — only
        # the .delta strings concatenate onto stdout
        rsx = runner.invoke(app, ["harness", "respond", "hi", "--stream"])
        out["respond_stream_local_concat"] = rsx.exit_code == 0 and rsx.stdout == "local\n"
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
        # --fallback is repeatable and packs into the chain on both surfaces.
        rfb = runner.invoke(
            app,
            [
                "harness",
                "complete",
                "hi",
                "--backend",
                "hosted_k3",
                "--fallback",
                "byok",
            ],
        )
        out["complete_fallback_flags_forward"] = rfb.exit_code == 0 and fake.complete_calls[-1].get(
            "fallbacks"
        ) == ["byok"]
        rfb_s = runner.invoke(
            app,
            [
                "harness",
                "complete",
                "hi",
                "--stream",
                "--backend",
                "hosted_k3",
                "--fallback",
                "local_fx1",
                "--fallback",
                "byok",
            ],
        )
        out["stream_fallback_flags_forward"] = rfb_s.exit_code == 0 and fake.stream_calls[-1].get(
            "fallbacks"
        ) == ["local_fx1", "byok"]
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
            rb_f = runner.invoke(
                app,
                [
                    "harness",
                    "batch",
                    str(pf),
                    "--backend",
                    "hosted_k3",
                    "--fallback",
                    "byok",
                ],
            )
            out["batch_fallback_flags_forward"] = rb_f.exit_code == 0 and fake.batch_calls[-1].get(
                "fallbacks"
            ) == ["byok"]
            rb_s = runner.invoke(
                app,
                [
                    "harness",
                    "batch",
                    str(pf),
                    "--backend",
                    "byok",
                    "--temperature",
                    "0.4",
                    "--seed",
                    "11",
                ],
            )
            out["batch_sampling_flags_forward"] = (
                rb_s.exit_code == 0
                and fake.batch_calls[-1].get("temperature") == 0.4
                and fake.batch_calls[-1].get("seed") == 11
            )

        # decode flags forward through both complete surfaces; unpinned calls
        # send nothing (the surface resolves the temperature=0.0 default).
        rsp = runner.invoke(
            app,
            [
                "harness",
                "complete",
                "hi",
                "--backend",
                "byok",
                "--temperature",
                "0.6",
                "--top-p",
                "0.8",
                "--max-tokens",
                "33",
                "--seed",
                "9",
            ],
        )
        rsp_last = fake.complete_calls[-1]
        out["complete_sampling_flags_forward"] = (
            rsp.exit_code == 0
            and rsp_last.get("temperature") == 0.6
            and rsp_last.get("top_p") == 0.8
            and rsp_last.get("max_tokens") == 33
            and rsp_last.get("seed") == 9
        )
        rsp_s = runner.invoke(
            app,
            [
                "harness",
                "complete",
                "hi",
                "--stream",
                "--backend",
                "byok",
                "--temperature",
                "0.6",
                "--seed",
                "9",
            ],
        )
        out["stream_sampling_flags_forward"] = (
            rsp_s.exit_code == 0
            and fake.stream_calls[-1].get("temperature") == 0.6
            and fake.stream_calls[-1].get("seed") == 9
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
            self.last_diff: tuple[str, str] | None = None
            self.last_ft_create: dict[str, Any] | None = None
            self.last_ft_job: str | None = None
            self.last_ft_query: dict[str, Any] | None = None
            self.last_upload: dict[str, Any] | None = None
            self.last_file_id: str | None = None
            self.last_batch_create: dict[str, Any] | None = None
            self.last_batch_id: str | None = None
            self.last_wait_kw: dict[str, Any] | None = None

        def complete(self, messages: Any, **kw: Any) -> CompletionResult:
            return CompletionResult(backend="byok", model="remote-v0", content="remote-text")

        def commands(self, role: Any = None) -> list[str]:
            self.last_ft_query = {"commands_role": role}
            return ["cmd-a"] if role is None else []

        def verify_receipts(self, receipts: list[dict[str, Any]]) -> Any:
            from fx1.sdk import ReceiptVerdict

            return tuple(
                ReceiptVerdict(
                    valid=i == 0,
                    path="<cli>",
                    errors=() if i == 0 else ("bad",),
                    warnings=(),
                    schema_tag="x",
                    kind="k",
                    verdict="ok" if i == 0 else "invalid",
                    digest_convention=None,
                )
                for i, _r in enumerate(receipts)
            )

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

        def job_receipt(self, job_id: str) -> dict[str, Any]:
            self.last_job = job_id
            return {
                "kind": "fx1_job_record",
                "schema": "fx1_job_record.v1",
                "record": {"job_id": job_id, "status": "succeeded"},
                "receipt_sha256": "0" * 64,
            }

        def list_jobs(self, **kw: Any) -> dict[str, Any]:
            self.last_jobs_query = dict(kw)
            return {
                "jobs": [{"job_id": "job-xyz", "status": "succeeded", "result": None}],
                "total": 1,
            }

        def cancel_job(self, job_id: str) -> dict[str, Any]:
            self.last_job = job_id
            return {"job_id": job_id, "status": "cancelled", "result": None}

        def diff_evals(self, base_id: str, candidate_id: str) -> dict[str, Any]:
            self.last_diff = (base_id, candidate_id)
            return {
                "object": "eval_diff",
                "base_eval_id": base_id,
                "candidate_eval_id": candidate_id,
                "comparable": True,
                "verdict": "unchanged",
            }

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

        def upload_file(self, content: bytes, **kw: Any) -> dict[str, Any]:
            self.last_upload = {
                "purpose": kw.get("purpose"),
                "filename": kw.get("filename"),
                "n_bytes": len(content),
            }
            return {
                "id": "file-ft",
                "object": "file",
                "purpose": kw.get("purpose"),
                "filename": kw.get("filename"),
                "bytes": len(content),
            }

        def create_finetune_job(self, **kw: Any) -> dict[str, Any]:
            self.last_ft_create = dict(kw)
            return {
                "id": "ftjob-x",
                "object": "fine_tuning.job",
                "status": "queued",
                "model": kw.get("model"),
            }

        def wait_finetune_job(self, job_id: str, **kw: Any) -> dict[str, Any]:
            self.last_ft_job = job_id
            self.last_wait_kw = dict(kw)
            return {
                "id": job_id,
                "object": "fine_tuning.job",
                "status": "succeeded",
                "fine_tuned_model": "ft:fx1:x:000000000000",
            }

        def wait_eval(self, eval_id: str, **kw: Any) -> dict[str, Any]:
            self.last_job = eval_id
            self.last_wait_kw = dict(kw)
            return {
                "eval_id": eval_id,
                "suite": "capability",
                "status": "succeeded",
                "report": {"passed": 2},
            }

        def eval_receipt(self, eval_id: str) -> dict[str, Any]:
            self.last_job = eval_id
            return {
                "schema": "fx1_eval_record.v1",
                "eval_id": eval_id,
                "receipt_sha256": "ab" * 32,
            }

        def finetune_jobs(self, **kw: Any) -> dict[str, Any]:
            self.last_ft_query = dict(kw)
            return {
                "object": "list",
                "data": [{"id": "ftjob-x", "object": "fine_tuning.job", "status": "succeeded"}],
                "has_more": False,
            }

        def finetune_job(self, job_id: str) -> dict[str, Any]:
            self.last_ft_job = job_id
            return {"id": job_id, "object": "fine_tuning.job", "status": "running"}

        def finetune_job_events(self, job_id: str, **kw: Any) -> dict[str, Any]:
            self.last_ft_job = job_id
            self.last_ft_query = dict(kw)
            return {
                "object": "list",
                "data": [{"id": "ftev-1", "object": "fine_tuning.job.event", "message": "m"}],
                "has_more": False,
            }

        def cancel_finetune_job(self, job_id: str) -> dict[str, Any]:
            self.last_ft_job = job_id
            return {"id": job_id, "object": "fine_tuning.job", "status": "cancelled"}

        def finetune_job_checkpoints(self, job_id: str, **kw: Any) -> dict[str, Any]:
            self.last_ft_job = job_id
            self.last_ft_query = dict(kw)
            return {
                "object": "list",
                "data": [
                    {
                        "id": "ftckpt-1",
                        "object": "fine_tuning.job.checkpoint",
                        "fine_tuned_model_checkpoint": "ft:fx1:x",
                    }
                ],
                "has_more": False,
            }

        def files(self) -> list[dict[str, Any]]:
            return [{"id": "file-1", "object": "file", "purpose": "batch"}]

        def file_content(self, file_id: str) -> bytes:
            self.last_file_id = file_id
            return b'{"custom_id":"r1"}\n'

        def delete_file(self, file_id: str) -> dict[str, Any]:
            self.last_file_id = file_id
            return {"id": file_id, "object": "file", "deleted": True}

        def create_batch(self, input_file_id: str, **kw: Any) -> dict[str, Any]:
            rec = dict(kw)
            rec["input_file_id"] = input_file_id
            self.last_batch_create = rec
            return {
                "id": "batch_x",
                "object": "batch",
                "status": "validating",
                "input_file_id": input_file_id,
            }

        def wait_batch(self, batch_id: str, **kw: Any) -> dict[str, Any]:
            self.last_batch_id = batch_id
            self.last_wait_kw = dict(kw)
            return {
                "id": batch_id,
                "object": "batch",
                "status": "completed",
                "output_file_id": "file-out",
            }

        def batches(self, **kw: Any) -> dict[str, Any]:
            self.last_ft_query = dict(kw)
            return {
                "object": "list",
                "data": [{"id": "batch_x", "object": "batch", "status": "completed"}],
                "has_more": False,
            }

        def batch(self, batch_id: str) -> dict[str, Any]:
            self.last_batch_id = batch_id
            return {
                "id": batch_id,
                "object": "batch",
                "status": "completed",
                "output_file_id": "file-out",
            }

        def cancel_batch(self, batch_id: str) -> dict[str, Any]:
            self.last_batch_id = batch_id
            return {"id": batch_id, "object": "batch", "status": "cancelling"}

        def list_models(self) -> dict[str, Any]:
            return {
                "object": "list",
                "data": [{"id": "fx1", "object": "model", "created": 1, "owned_by": "fx1"}],
            }

        def retrieve_model(self, model_id: str) -> dict[str, Any]:
            self.last_ft_query = {"model": model_id}
            return {"id": model_id, "object": "model", "created": 1, "owned_by": "fx1"}

        def delete_model(self, model_id: str) -> dict[str, Any]:
            self.last_ft_query = {"delete": model_id}
            return {"id": model_id, "object": "model", "deleted": True}

        def responses_create(
            self,
            input: Any,
            **kw: Any,  # noqa: A002
        ) -> tuple[dict[str, Any], None]:
            self.last_ft_query = {"input": input, **kw}
            return (
                {
                    "id": "resp_x",
                    "object": "response",
                    "model": kw.get("model", "fx1"),
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hi"}],
                        }
                    ],
                },
                None,
            )

        def responses_create_stream(
            self,
            input: Any,
            **kw: Any,  # noqa: A002
        ) -> tuple[list[dict[str, Any]], str]:
            self.last_ft_query = {"input": input, "stream": True, **kw}
            return (
                [
                    {"type": "response.output_text.delta", "delta": "rem"},
                    {"type": "response.output_text.delta", "delta": "ote"},
                    {"type": "response.completed", "response": {"id": "resp_x"}},
                ],
                "cid-stream",
            )

        def embeddings_create(
            self,
            input: Any,
            **kw: Any,  # noqa: A002
        ) -> tuple[dict[str, Any], None]:
            self.last_ft_query = {"input": input, **kw}
            return (
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                    "model": kw.get("model", "fx1"),
                },
                None,
            )

        def moderate(self, input: Any) -> dict[str, Any]:  # noqa: A002
            self.last_ft_query = {"input": input}
            return {
                "id": "modr-x",
                "model": "fx1-honesty-gate",
                "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
            }

        def retrieve_chat_completion(self, completion_id: str) -> dict[str, Any]:
            self.last_ft_query = {"chat_get": completion_id}
            return {"id": completion_id, "object": "chat.completion"}

        def delete_chat_completion(self, completion_id: str) -> dict[str, Any]:
            self.last_ft_query = {"chat_delete": completion_id}
            return {
                "id": completion_id,
                "object": "chat.completion.deleted",
                "deleted": True,
            }

        def retrieve_response(self, response_id: str) -> dict[str, Any]:
            self.last_ft_query = {"resp_get": response_id}
            return {"id": response_id, "object": "response"}

        def delete_response(self, response_id: str) -> dict[str, Any]:
            self.last_ft_query = {"resp_delete": response_id}
            return {
                "id": response_id,
                "object": "response.deleted",
                "deleted": True,
            }

        def list_chat_completions(self, **kw: Any) -> dict[str, Any]:
            self.last_ft_query = {"chat_list": True, **kw}
            return {
                "object": "list",
                "data": [{"id": "chatcmpl-x", "object": "chat.completion"}],
                "first_id": "chatcmpl-x",
                "last_id": "chatcmpl-x",
                "has_more": False,
            }

        def chat_completion_messages(self, completion_id: str, **kw: Any) -> dict[str, Any]:
            self.last_ft_query = {"chat_messages": completion_id, **kw}
            return {
                "object": "list",
                "data": [{"id": "msg_x", "role": "user", "content": "hi"}],
                "first_id": "msg_x",
                "last_id": "msg_x",
                "has_more": False,
            }

        def response_input_items(self, response_id: str, **kw: Any) -> dict[str, Any]:
            self.last_ft_query = {"input_items": response_id, **kw}
            return {
                "object": "list",
                "data": [{"id": "msg_y", "type": "message", "role": "user"}],
                "first_id": "msg_y",
                "last_id": "msg_y",
                "has_more": False,
            }

        def score(self, input: Any) -> list[dict[str, Any]]:  # noqa: A002
            self.last_ft_query = {"score": input}
            items = input if isinstance(input, list) else [input]
            return [
                {
                    "object": "score",
                    "index": i,
                    "total": 0.0,
                    "components": {},
                    "violations": [],
                }
                for i, _t in enumerate(items)
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
        red = runner.invoke(
            app, ["harness", "eval-diff", "ev-a", "ev-b", "--remote", "http://h.test"]
        )
        out["remote_evaldiff_json"] = (
            red.exit_code == 0
            and json.loads(red.stdout).get("object") == "eval_diff"
            and remotes[-1].last_diff == ("ev-a", "ev-b")
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
    rjr_local = runner.invoke(app, ["harness", "job", "j-9", "--receipt"])
    out["job_receipt_local_refused"] = rjr_local.exit_code == 2 and "--remote" in rjr_local.output
    red_local = runner.invoke(app, ["harness", "eval-diff", "ev-a", "ev-b"])
    out["evaldiff_local_refused"] = red_local.exit_code == 2 and "--remote" in red_local.output
    rfj_local = runner.invoke(app, ["harness", "ft-jobs"])
    out["ftjobs_local_refused"] = rfj_local.exit_code == 2 and "--remote" in rfj_local.output
    rfs_local = runner.invoke(app, ["harness", "ft-status", "ftjob-x"])
    out["ftstatus_local_refused"] = rfs_local.exit_code == 2 and "--remote" in rfs_local.output
    rfe_local = runner.invoke(app, ["harness", "ft-events", "ftjob-x"])
    out["ftevents_local_refused"] = rfe_local.exit_code == 2 and "--remote" in rfe_local.output
    rfc_local = runner.invoke(app, ["harness", "ft-cancel", "ftjob-x"])
    out["ftcancel_local_refused"] = rfc_local.exit_code == 2 and "--remote" in rfc_local.output
    rew_local = runner.invoke(app, ["harness", "eval-wait", "ev-x"])
    out["evalwait_local_refused"] = rew_local.exit_code == 2 and "--remote" in rew_local.output
    rfw_local = runner.invoke(app, ["harness", "ft-wait", "ftjob-x"])
    out["ftwait_local_refused"] = rfw_local.exit_code == 2 and "--remote" in rfw_local.output
    # ft-create local runs the in-process SDK twin synchronously — a
    # malformed corpus fails validation before any job exists (exit 2).
    import tempfile  # noqa: PLC0415
    from pathlib import Path as _Path2  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as ftd2:
        bad_corpus = _Path2(ftd2) / "bad.jsonl"
        bad_corpus.write_bytes(b"not jsonl\n")
        out["ft_create_local_bad_corpus_2"] = (
            runner.invoke(app, ["harness", "ft-create", str(bad_corpus)]).exit_code == 2
        )
        missing = _Path2(ftd2) / "missing.jsonl"
        out["ft_create_local_missing_2"] = (
            runner.invoke(app, ["harness", "ft-create", str(missing)]).exit_code == 2
        )

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
        rjr = runner.invoke(
            app,
            ["harness", "job", "j-9", "--remote", "http://h.test", "--receipt"],
        )
        out["cli_job_receipt_json"] = (
            rjr.exit_code == 0
            and json.loads(rjr.stdout)["record"]["job_id"] == "j-9"
            and json.loads(rjr.stdout)["schema"] == "fx1_job_record.v1"
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

        # fine-tuning surface on the wire: corpus upload (purpose=fine-tune),
        # submit with flags, wait, list, status, events, cancel — all remote.
        with tempfile.TemporaryDirectory() as ftd:
            cpath = _Path(ftd) / "corpus.jsonl"
            cpath.write_bytes(
                b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
            )
            rft = runner.invoke(
                app,
                [
                    "harness",
                    "ft-create",
                    str(cpath),
                    "--remote",
                    "http://h.test",
                    "--suffix",
                    "pp",
                    "--epochs",
                    "3",
                    "--seed",
                    "11",
                ],
            )
            out["remote_ft_create_json"] = (
                rft.exit_code == 0
                and json.loads(rft.stdout)["status"] == "succeeded"
                and remotes[-1].last_upload
                == {
                    "purpose": "fine-tune",
                    "filename": "corpus.jsonl",
                    "n_bytes": len(cpath.read_bytes()),
                }
                and remotes[-1].last_ft_create
                == {
                    "model": "fx1",
                    "training_file": "file-ft",
                    "hyperparameters": {"n_epochs": 3},
                    "suffix": "pp",
                    "validation_file": None,
                    "seed": 11,
                    "callback_url": None,
                    "callback_secret": None,
                }
                and remotes[-1].last_ft_job == "ftjob-x"
            )
            # --callback-url/--callback-secret flow through to the client
            rft_cb = runner.invoke(
                app,
                [
                    "harness",
                    "ft-create",
                    str(cpath),
                    "--remote",
                    "http://h.test",
                    "--no-wait",
                    "--callback-url",
                    "https://hooks.test/ft",
                    "--callback-secret",
                    "whsec-cli",
                ],
            )
            out["remote_ft_create_callback_kwargs"] = (
                rft_cb.exit_code == 0
                and remotes[-1].last_ft_create is not None
                and remotes[-1].last_ft_create.get("callback_url") == "https://hooks.test/ft"
                and remotes[-1].last_ft_create.get("callback_secret") == "whsec-cli"
            )
            rft_nw = runner.invoke(
                app,
                [
                    "harness",
                    "ft-create",
                    str(cpath),
                    "--remote",
                    "http://h.test",
                    "--no-wait",
                ],
            )
            out["remote_ft_create_nowait"] = (
                rft_nw.exit_code == 0
                and json.loads(rft_nw.stdout)["status"] == "queued"
                and remotes[-1].last_ft_job is None  # wait never ran
            )
            out["ft_create_missing_file_2"] = (
                runner.invoke(
                    app,
                    [
                        "harness",
                        "ft-create",
                        str(_Path(ftd) / "nope.jsonl"),
                        "--remote",
                        "http://h.test",
                    ],
                ).exit_code
                == 2
            )
            # a file-* positional skips the upload — the id goes verbatim.
            rft_id = runner.invoke(
                app,
                [
                    "harness",
                    "ft-create",
                    "file-preloaded",
                    "--remote",
                    "http://h.test",
                    "--no-wait",
                    "--validation-file",
                    "file-val",
                ],
            )
            out["remote_ft_create_fileid"] = (
                rft_id.exit_code == 0
                and remotes[-1].last_upload is None
                and (remotes[-1].last_ft_create or {}).get("training_file") == "file-preloaded"
                and (remotes[-1].last_ft_create or {}).get("validation_file") == "file-val"
            )
            # in-process, file-* ids are an arg fault — the twin needs bytes.
            out["ft_create_fileid_inprocess_2"] = (
                runner.invoke(app, ["harness", "ft-create", "file-x"]).exit_code == 2
            )
        rfj = runner.invoke(
            app,
            [
                "harness",
                "ft-jobs",
                "--remote",
                "http://h.test",
                "--limit",
                "7",
                "--after",
                "ftjob-a",
            ],
        )
        out["remote_ft_jobs_json"] = (
            rfj.exit_code == 0
            and json.loads(rfj.stdout)["data"][0]["id"] == "ftjob-x"
            and remotes[-1].last_ft_query == {"limit": 7, "after": "ftjob-a"}
        )
        rfs = runner.invoke(app, ["harness", "ft-status", "ftjob-x", "--remote", "http://h.test"])
        out["remote_ft_status_json"] = (
            rfs.exit_code == 0
            and json.loads(rfs.stdout)["status"] == "running"
            and remotes[-1].last_ft_job == "ftjob-x"
        )
        rfe = runner.invoke(
            app,
            ["harness", "ft-events", "ftjob-x", "--remote", "http://h.test", "--limit", "9"],
        )
        out["remote_ft_events_json"] = (
            rfe.exit_code == 0
            and json.loads(rfe.stdout)["data"][0]["object"] == "fine_tuning.job.event"
            and remotes[-1].last_ft_query == {"limit": 9}
        )
        rfc = runner.invoke(app, ["harness", "ft-cancel", "ftjob-x", "--remote", "http://h.test"])
        out["remote_ft_cancel_json"] = (
            rfc.exit_code == 0
            and json.loads(rfc.stdout)["status"] == "cancelled"
            and remotes[-1].last_ft_job == "ftjob-x"
        )

        # the *-wait twins re-attach to a --no-wait submit: poll kwargs
        # forward, the terminal record prints, eval-wait --receipt follows.
        rew = runner.invoke(
            app,
            [
                "harness",
                "eval-wait",
                "ev-9",
                "--poll",
                "0.2",
                "--wait-timeout",
                "5",
                "--receipt",
                "--remote",
                "http://h.test",
            ],
        )
        out["remote_eval_wait"] = (
            rew.exit_code == 0
            and '"status": "succeeded"' in rew.stdout
            and '"schema": "fx1_eval_record.v1"' in rew.stdout
            and remotes[-1].last_wait_kw == {"poll_s": 0.2, "timeout_s": 5.0}
        )
        rfw = runner.invoke(app, ["harness", "ft-wait", "ftjob-9", "--remote", "http://h.test"])
        out["remote_ft_wait"] = (
            rfw.exit_code == 0
            and json.loads(rfw.stdout)["fine_tuned_model"] == "ft:fx1:x:000000000000"
            and remotes[-1].last_ft_job == "ftjob-9"
            and remotes[-1].last_wait_kw == {"poll_s": 0.5, "timeout_s": None}
        )
        rbw = runner.invoke(app, ["harness", "batch-wait", "batch_z", "--remote", "http://h.test"])
        out["remote_batch_wait"] = (
            rbw.exit_code == 0
            and json.loads(rbw.stdout)["output_file_id"] == "file-out"
            and remotes[-1].last_batch_id == "batch_z"
        )

        # /v1/files + /v1/batches wire family
        with tempfile.TemporaryDirectory() as bfd:
            blines = _Path(bfd) / "in.jsonl"
            blines.write_bytes(
                b'{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{}}\n'
            )
            out["remote_files_list"] = (
                runner.invoke(app, ["harness", "files", "--remote", "http://h.test"]).exit_code == 0
            )
            rfu = runner.invoke(
                app,
                [
                    "harness",
                    "file-upload",
                    str(blines),
                    "--purpose",
                    "batch",
                    "--remote",
                    "http://h.test",
                ],
            )
            out["remote_file_upload_json"] = (
                rfu.exit_code == 0
                and json.loads(rfu.stdout)["id"] == "file-ft"
                and remotes[-1].last_upload
                == {
                    "purpose": "batch",
                    "filename": "in.jsonl",
                    "n_bytes": len(blines.read_bytes()),
                }
            )
            rfc_out = runner.invoke(
                app, ["harness", "file-content", "file-9", "--remote", "http://h.test"]
            )
            out["remote_file_content_stdout"] = (
                rfc_out.exit_code == 0
                and rfc_out.stdout.startswith('{"custom_id"')
                and remotes[-1].last_file_id == "file-9"
            )
            out["remote_file_delete_json"] = (
                runner.invoke(
                    app, ["harness", "file-delete", "file-9", "--remote", "http://h.test"]
                ).exit_code
                == 0
                and remotes[-1].last_file_id == "file-9"
            )
            # batch-submit: upload (purpose=batch) → create → wait → print
            rbsub = runner.invoke(
                app,
                [
                    "harness",
                    "batch-submit",
                    str(blines),
                    "--remote",
                    "http://h.test",
                    "--metadata",
                    '{"k":"v"}',
                    "--callback-url",
                    "https://hooks.test/b",
                    "--callback-secret",
                    "whsec-b",
                ],
            )
            out["remote_batch_submit_terminal"] = (
                rbsub.exit_code == 0
                and json.loads(rbsub.stdout)["status"] == "completed"
                and (remotes[-1].last_upload or {}).get("purpose") == "batch"
                and remotes[-1].last_batch_create
                == {
                    "endpoint": "/v1/chat/completions",
                    "metadata": {"k": "v"},
                    "idempotency_key": None,
                    "callback_url": "https://hooks.test/b",
                    "callback_secret": "whsec-b",
                    "input_file_id": "file-ft",
                }
                and remotes[-1].last_batch_id == "batch_x"
            )
            out["remote_batch_submit_bad_meta_2"] = (
                runner.invoke(
                    app,
                    [
                        "harness",
                        "batch-submit",
                        str(blines),
                        "--remote",
                        "http://h.test",
                        "--metadata",
                        "[1]",
                    ],
                ).exit_code
                == 2
            )
            out["remote_batches_list"] = runner.invoke(
                app, ["harness", "batches", "--remote", "http://h.test"]
            ).exit_code == 0 and remotes[-1].last_ft_query == {"limit": 20, "after": None}
            out["remote_batch_status_json"] = (
                runner.invoke(
                    app,
                    ["harness", "batch-status", "batch_x", "--remote", "http://h.test"],
                ).exit_code
                == 0
                and remotes[-1].last_batch_id == "batch_x"
            )
            out["remote_batch_cancel_json"] = (
                json.loads(
                    runner.invoke(
                        app,
                        ["harness", "batch-cancel", "batch_x", "--remote", "http://h.test"],
                    ).stdout
                )["status"]
                == "cancelling"
            )
            # batch-output: batch record → output_file_id → file bytes
            rbo = runner.invoke(
                app,
                [
                    "harness",
                    "batch-output",
                    "batch_x",
                    "--remote",
                    "http://h.test",
                    "--out",
                    str(_Path(bfd) / "out.jsonl"),
                ],
            )
            out["remote_batch_output_writes_file"] = (
                rbo.exit_code == 0
                and (_Path(bfd) / "out.jsonl").read_bytes().startswith(b'{"custom_id"')
                and remotes[-1].last_file_id == "file-out"
            )

        # a batch with no output_file_id exits 2, not a traceback
        class _NoOutRemote(_FakeRemote):
            def batch(self, batch_id: str) -> dict[str, Any]:
                return {"id": batch_id, "object": "batch", "status": "failed"}

        def _mk_noout(*a: Any, **kw: Any) -> _NoOutRemote:
            nr = _NoOutRemote(*a, **kw)
            remotes.append(nr)
            return nr

        with patch("fx1.serve.client.HarnessClient", side_effect=_mk_noout):
            out["remote_batch_output_missing_2"] = (
                runner.invoke(
                    app,
                    ["harness", "batch-output", "batch_x", "--remote", "http://h.test"],
                ).exit_code
                == 2
            )

        out["remote_models_list"] = (
            json.loads(
                runner.invoke(app, ["harness", "models", "--remote", "http://h.test"]).stdout
            )
            .get("data", [{}])[0]
            .get("id")
            == "fx1"
        )
        out["remote_model_retrieve"] = (
            json.loads(
                runner.invoke(
                    app, ["harness", "model", "ft:fx1-x", "--remote", "http://h.test"]
                ).stdout
            ).get("id")
            == "ft:fx1-x"
        )
        out["remote_model_delete"] = (
            json.loads(
                runner.invoke(
                    app, ["harness", "model-delete", "ft:fx1-x", "--remote", "http://h.test"]
                ).stdout
            ).get("deleted")
            is True
        )
        out["model_delete_inproc_400"] = (
            runner.invoke(app, ["harness", "model-delete", "fx1"]).exit_code == 2
        )
        # ft-checkpoints is wire-only — --remote forwards job id + paging.
        out["remote_ft_checkpoints"] = (
            json.loads(
                runner.invoke(
                    app,
                    [
                        "harness",
                        "ft-checkpoints",
                        "ftjob-x",
                        "--remote",
                        "http://h.test",
                        "--limit",
                        "5",
                    ],
                ).stdout
            )["data"][0]["object"]
            == "fine_tuning.job.checkpoint"
            and remotes[-1].last_ft_job == "ftjob-x"
            and remotes[-1].last_ft_query == {"limit": 5, "after": None}
        )
        out["ft_checkpoints_inproc_exit2"] = (
            runner.invoke(app, ["harness", "ft-checkpoints", "ftjob-x"]).exit_code == 2
        )

        _rr = runner.invoke(
            app,
            [
                "harness",
                "respond",
                "say hi",
                "--model",
                "fx1",
                "--backend",
                "local_fx1",
                "--checkpoint-dir",
                "ckpt-x",
                "--metadata",
                '{"k":"v"}',
                "--format",
                '{"type":"json_object"}',
                "--remote",
                "http://h.test",
            ],
        )
        out["remote_respond_kwargs"] = (
            _rr.exit_code == 0
            and (remotes[-1].last_ft_query or {}).get("text_format") == {"type": "json_object"}
            and (remotes[-1].last_ft_query or {}).get("metadata") == {"k": "v"}
            and json.loads(_rr.stdout).get("id") == "resp_x"
        )
        # respond --stream remote-side: bare payload dicts — the deltas
        # concatenate and the flag kwargs forward verbatim
        _rsr = runner.invoke(
            app,
            [
                "harness",
                "respond",
                "say hi",
                "--stream",
                "--format",
                '{"type":"json_object"}',
                "--remote",
                "http://h.test",
            ],
        )
        out["remote_respond_stream"] = (
            _rsr.exit_code == 0
            and _rsr.stdout == "remote\n"
            and (remotes[-1].last_ft_query or {}).get("stream") is True
            and (remotes[-1].last_ft_query or {}).get("text_format") == {"type": "json_object"}
        )
        out["remote_embed"] = json.loads(
            runner.invoke(
                app,
                ["harness", "embed", "a", "b", "--remote", "http://h.test"],
            ).stdout
        ).get("data", [{}])[0].get("embedding") == [0.1]
        out["remote_moderate"] = (
            json.loads(
                runner.invoke(
                    app, ["harness", "moderate", "hi", "--remote", "http://h.test"]
                ).stdout
            ).get("model")
            == "fx1-honesty-gate"
        )
        out["remote_stored_family"] = all(
            json.loads(
                runner.invoke(app, ["harness", name, "id-x", "--remote", "http://h.test"]).stdout
            ).get("id")
            == "id-x"
            for name in (
                "chat-get",
                "chat-delete",
                "response-get",
                "response-delete",
            )
        )
        # the stored-request subresources — --remote forwards id + paging
        out["remote_chat_messages"] = json.loads(
            runner.invoke(
                app,
                [
                    "harness",
                    "chat-messages",
                    "chatcmpl-x",
                    "--limit",
                    "5",
                    "--order",
                    "desc",
                    "--remote",
                    "http://h.test",
                ],
            ).stdout
        ).get("last_id") == "msg_x" and remotes[-1].last_ft_query == {
            "chat_messages": "chatcmpl-x",
            "limit": 5,
            "after": None,
            "before": None,
            "order": "desc",
        }
        out["remote_chat_list"] = json.loads(
            runner.invoke(
                app,
                [
                    "harness",
                    "chat-list",
                    "--model",
                    "fx1",
                    "--metadata",
                    "lane=lp",
                    "--limit",
                    "3",
                    "--remote",
                    "http://h.test",
                ],
            ).stdout
        ).get("first_id") == "chatcmpl-x" and remotes[-1].last_ft_query == {
            "chat_list": True,
            "model": "fx1",
            "metadata": {"lane": "lp"},
            "limit": 3,
            "after": None,
            "before": None,
            "order": "asc",
        }
        out["remote_response_input_items"] = json.loads(
            runner.invoke(
                app,
                ["harness", "response-input-items", "resp_x", "--remote", "http://h.test"],
            ).stdout
        ).get("first_id") == "msg_y" and remotes[-1].last_ft_query == {
            "input_items": "resp_x",
            "limit": 20,
            "after": None,
            "before": None,
            "order": "asc",
        }
        _rs = json.loads(
            runner.invoke(app, ["harness", "score", "a", "b", "--remote", "http://h.test"]).stdout
        )
        out["remote_score"] = len(_rs) == 2 and _rs[0]["object"] == "score"
        out["remote_commands_role"] = (
            json.loads(
                runner.invoke(
                    app,
                    [
                        "harness",
                        "commands",
                        "--role",
                        "evaluation",
                        "--remote",
                        "http://h.test",
                    ],
                ).stdout
            )["commands"]
            == []
        )
        # verify on a directory is one batch call on the wire; an invalid
        # file exits 1 without aborting the rest.
        with _tmpf.TemporaryDirectory() as _rd_s:
            _rd = _P(_rd_s)
            (_rd / "a.json").write_text('{"x": 1}')
            (_rd / "b.json").write_text('{"y": 2}')
            _dv = runner.invoke(app, ["harness", "verify", str(_rd), "--remote", "http://h.test"])
        out["remote_verify_dir"] = _dv.exit_code == 1 and json.loads(_dv.stdout)["files"] == 2

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
            "SYNTHETIC-labeled, doctor emits presence-only JSON, and the "
            "completion log reads cleanly (empty window + missing-id exit "
            "2 on both the record and its --receipt export); harness job "
            "--receipt prints the sealed fx1_job_record.v1 doc remote-side "
            "and refuses without --remote (exit 2). respond --stream "
            "concatenates Responses delta frames on both surfaces "
            "(in-process SDK pairs and remote payload dicts), forwarding "
            "flag kwargs verbatim. ft-create/ft-jobs/"
            "ft-status/ft-events/ft-cancel hold the OpenAI job grammar "
            "remote-side (upload → submit → wait, --no-wait prints the "
            "queued record) and refuse locally without --remote. "
            "eval-wait/ft-wait/batch-wait re-attach to a --no-wait submit: "
            "poll kwargs reach the client verbatim, the terminal record "
            "prints, and eval-wait --receipt follows with the sealed doc; "
            "all three refuse locally without --remote (exit 2). "
            "Flagged wart: "
            "the command is registered as 'maskedaEval'."
            if ok
            else f"CLI AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
