"""sdk_audit — adversarial probes on ``fx1.sdk.Fx1Harness``.

The SDK is the in-process twin of ``fx1.serve.api``: fx-1 drives the same
registry, the same three backends, the same honesty gate, and the same
receipt verifier without a socket. If the two surfaces drift — a command
listed only over HTTP, an error class that changes meaning in-process —
the harness contract forks and receipts stop meaning the same thing on
both paths. These probes pin the parity.

Pinned contract:

- *Registry parity* — ``commands()`` returns exactly ``HARNESS_REGISTRY``;
  ``role=`` filters by ``HarnessRole``; unknown names raise ``KeyError``
  on ``run`` and ``command`` (the 404-class).
- *Run path* — the injectable runner sees ``dipcatcher``-less argv from
  the registry; ``config`` and ``extra_args --config`` are contained to
  ``configs/``; the result carries exit_code/stdout/stderr/ok.
- *Completion path* — ``local_fx1`` requires a checkpoint (``ValueError``
  — the 422-class); ``checkpoint_dir`` is rejected for other backends;
  ``BackendNotConfiguredError`` propagates untransformed (the 503-class);
  the honesty gate runs before the result returns (``Fx1HonestyError`` —
  the 502-class); ``backend.close()`` is always invoked, even on gate
  failure — spawned engines never leak; ``receipt_hashes`` append the
  evidence footer; ``model``/``backend`` echo what the backend resolved.
- *Receipt surface* — ``verify_receipt`` returns the full verifier shape;
  a sealed bench receipt verifies, a tampered digest fails closed.
- *Health* — presence booleans only: no env values, no paths leak; the
  ``local_fx1`` flag requires a card'd checkpoint AND a serve env.

Sealed ``sdk_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fx1.serve.backends import SamplingParams

__all__ = ["sdk_audit", "sdk_audit_bench"]


class _FakeBackend:
    """Records close() and returns a canned, honesty-clean completion."""

    def __init__(self, content: str = "clean answer") -> None:
        self._model = "fake-0"
        self.closed = 0
        self.calls = 0
        self._content = content
        self.seen_messages: list[dict[str, str]] = []

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        self.calls += 1
        self.seen_messages = list(messages)
        return self._content

    def stream(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> Any:
        yield self._content[: len(self._content) // 2]
        yield self._content[len(self._content) // 2 :]

    def close(self) -> None:
        self.closed += 1


class _NonStreamingBackend:
    """complete-only backend — the stream contract must fail closed."""

    def __init__(self) -> None:
        self.closed = 0

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        return "clean answer"

    def close(self) -> None:
        self.closed += 1


def sdk_audit() -> dict[str, bool]:
    """Probe the SDK contract; every key must be True."""
    from fx1.harness import HARNESS_REGISTRY, Harness, HarnessRole
    from fx1.sdk import Fx1Harness
    from fx1.serve.backends import BackendNotConfiguredError

    out: dict[str, bool] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:  # noqa: BLE001 - probe records the class
            return type(e).__name__

    # ---- registry parity -------------------------------------------------
    sdk = Fx1Harness()
    out["registry_full"] = sdk.commands() == [c.name for c in HARNESS_REGISTRY]
    out["registry_role_filter"] = sdk.commands(HarnessRole.VERIFICATION) == [
        c.name for c in HARNESS_REGISTRY if c.role == HarnessRole.VERIFICATION
    ]
    out["registry_nonempty"] = len(sdk.commands()) > 0
    cmd = sdk.command("doctor")
    out["command_shape"] = cmd.argv == ["doctor"] and cmd.role == HarnessRole.VERIFICATION
    out["unknown_command_404"] = _raises(lambda: sdk.command("not-a-command")) == "KeyError"
    out["unknown_run_404"] = _raises(lambda: sdk.run("not-a-command")) == "KeyError"

    # ---- run path ----------------------------------------------------------
    seen: dict[str, Any] = {}

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen["argv"] = list(argv)
        seen["timeout_s"] = timeout_s
        return 7, "out-text", "err-text"

    sdk2 = Fx1Harness(harness=Harness(runner=fake_runner))
    res = sdk2.run("doctor")
    out["run_executes_registry_argv"] = seen["argv"] == ["doctor"]
    out["run_result_shape"] = (
        res.exit_code == 7 and res.stdout == "out-text" and res.stderr == "err-text"
    )
    out["run_ok_flag"] = res.ok is False
    out["run_timeout_bound"] = seen["timeout_s"] == sdk2.command("doctor").timeout_s
    out["run_extra_args"] = sdk2.run("doctor", ["--foo", "1"]).exit_code == 7 and seen["argv"] == [
        "doctor",
        "--foo",
        "1",
    ]

    with tempfile.TemporaryDirectory() as td:
        configs = Path(td) / "configs"
        configs.mkdir()
        inside = configs / "ok.yml"
        inside.write_text("x: 1\n")
        cwd = Path.cwd()
        os.chdir(td)
        try:
            out["config_inside_ok"] = sdk2.run("doctor", config=inside).exit_code == 7
            outside = Path(td) / "evil.yml"
            outside.write_text("x: 1\n")
            out["config_escape_422"] = (
                _raises(lambda: sdk2.run("doctor", config=outside)) == "ValueError"
            )
            out["config_extra_args_escape_422"] = (
                _raises(lambda: sdk2.run("doctor", ["--config", str(outside)])) == "ValueError"
            )
            out["config_eq_escape_422"] = (
                _raises(lambda: sdk2.run("doctor", [f"--config={outside}"])) == "ValueError"
            )
        finally:
            os.chdir(cwd)

    # ---- completion path ---------------------------------------------------
    fake = _FakeBackend()
    sdk3 = Fx1Harness(backend_resolver=lambda *a, **k: fake)
    result = sdk3.complete([{"role": "user", "content": "hi"}], backend="byok")
    out["complete_runs_backend"] = result.content == "clean answer"
    out["complete_result_fields"] = result.backend == "byok" and result.model == "fake-0"
    out["complete_backend_closed"] = fake.closed == 1
    out["complete_messages_passthrough"] = fake.seen_messages == [{"role": "user", "content": "hi"}]
    cited = sdk3.complete(
        [{"role": "user", "content": "hi"}],
        backend="byok",
        receipt_hashes=["a" * 64],
    )
    out["cited_footer"] = "Evidence:" in cited.content and "`aaaaaaaaaaaaaaaa…`" in cited.content
    out["cited_hashes_echoed"] = cited.receipt_hashes == ("a" * 64,)

    # complete_many: one resolved backend serves the whole batch, in order
    shared = _FakeBackend()
    resolves = [0]

    def _resolve_once(*a: Any, **k: Any) -> _FakeBackend:
        resolves[0] += 1
        return shared

    sdk7 = Fx1Harness(backend_resolver=_resolve_once)
    batch = sdk7.complete_many(
        [[{"role": "user", "content": f"m{i}"}] for i in range(5)],
        backend="byok",
        max_workers=4,
    )
    out["batch_count_order"] = len(batch) == 5 and all(r.content == "clean answer" for r in batch)
    out["batch_shared_backend"] = resolves[0] == 1 and shared.calls == 5
    out["batch_closed_once"] = shared.closed == 1
    out["batch_result_fields"] = all(r.backend == "byok" and r.model == "fake-0" for r in batch)
    out["batch_empty_no_resolve"] = (
        sdk7.complete_many([], backend="byok") == [] and resolves[0] == 1
    )
    out["batch_workers_422"] = (
        _raises(
            lambda: sdk7.complete_many(
                [[{"role": "u", "content": "x"}]], backend="byok", max_workers=0
            )
        )
        == "ValueError"
    )
    out["batch_checkpoint_rejected"] = (
        _raises(
            lambda: sdk7.complete_many(
                [[{"role": "u", "content": "x"}]],
                backend="byok",
                checkpoint_dir="some-checkpoint",
            )
        )
        == "ValueError"
    )
    dirty_batch = _FakeBackend("total Sharpe 9.9 on NAV")
    sdk8 = Fx1Harness(backend_resolver=lambda *a, **k: dirty_batch)
    out["batch_gate_propagates"] = (
        _raises(lambda: sdk8.complete_many([[{"role": "u", "content": "x"}]], backend="byok"))
        == "Fx1HonestyError"
    )
    out["batch_closed_on_gate_fail"] = dirty_batch.closed == 1

    # stream_complete: buffered deltas, gated before they reach the caller
    stream_be = _FakeBackend()
    sdk9 = Fx1Harness(backend_resolver=lambda *a, **k: stream_be)
    chunks = sdk9.stream_complete([{"role": "user", "content": "hi"}], backend="byok")
    out["stream_chunks_ordered"] = "".join(chunks) == "clean answer"
    out["stream_backend_closed"] = stream_be.closed == 1
    cited_chunks = sdk9.stream_complete(
        [{"role": "u", "content": "x"}],
        backend="byok",
        receipt_hashes=["a" * 64],
    )
    out["stream_footer_cited"] = "Evidence:" in cited_chunks[-1]
    out["stream_closed_on_cited"] = stream_be.closed == 2
    dirty_stream = _FakeBackend("total Sharpe 9.9 on NAV")
    sdk10 = Fx1Harness(backend_resolver=lambda *a, **k: dirty_stream)
    out["stream_gate_propagates"] = (
        _raises(lambda: sdk10.stream_complete([{"role": "u", "content": "x"}], backend="byok"))
        == "Fx1HonestyError"
    )
    out["stream_closed_on_gate_fail"] = dirty_stream.closed == 1
    plain = _NonStreamingBackend()
    sdk11 = Fx1Harness(backend_resolver=lambda *a, **k: plain)
    out["stream_unsupported_501"] = (
        _raises(lambda: sdk11.stream_complete([{"role": "u", "content": "x"}], backend="byok"))
        == "NotImplementedError"
    )
    out["stream_unsupported_closed"] = plain.closed == 1
    out["stream_checkpoint_rejected"] = (
        _raises(
            lambda: sdk9.stream_complete(
                [{"role": "u", "content": "x"}],
                backend="byok",
                checkpoint_dir="some-checkpoint",
            )
        )
        == "ValueError"
    )

    out["checkpoint_rejected_nonlocal"] = (
        _raises(
            lambda: sdk3.complete(
                [{"role": "u", "content": "x"}],
                backend="byok",
                checkpoint_dir="some-checkpoint",
            )
        )
        == "ValueError"
    )
    env_ckpt = os.environ.pop("FX1_CHECKPOINT_DIR", None)
    try:
        out["local_needs_checkpoint"] = (
            _raises(lambda: sdk3.complete([{"role": "u", "content": "x"}], backend="local_fx1"))
            == "ValueError"
        )
    finally:
        if env_ckpt is not None:
            os.environ["FX1_CHECKPOINT_DIR"] = env_ckpt

    def _unconfigured(*a: Any, **k: Any) -> Any:
        raise BackendNotConfiguredError("no creds")

    sdk4 = Fx1Harness(backend_resolver=_unconfigured)
    out["unconfigured_503_class"] = (
        _raises(lambda: sdk4.complete([{"role": "u", "content": "x"}], backend="byok"))
        == "BackendNotConfiguredError"
    )

    def _unknown(*a: Any, **k: Any) -> Any:
        raise KeyError("bogus-backend")

    sdk5 = Fx1Harness(backend_resolver=_unknown)
    out["unknown_backend_404"] = (
        _raises(lambda: sdk5.complete([{"role": "u", "content": "x"}], backend="byok"))
        == "KeyError"
    )

    dirty = _FakeBackend("total Sharpe 4.2 on NAV")  # forbidden headline
    sdk6 = Fx1Harness(backend_resolver=lambda *a, **k: dirty)
    out["honesty_gate_502"] = (
        _raises(lambda: sdk6.complete([{"role": "u", "content": "x"}], backend="byok"))
        == "Fx1HonestyError"
    )
    out["closed_on_gate_fail"] = dirty.closed == 1

    # ---- receipts ------------------------------------------------------------
    from fx1.serve.byok_audit import byok_audit_bench

    good = byok_audit_bench()
    verdict = sdk.verify_receipt(good)
    out["verify_valid_receipt"] = verdict.valid is True
    out["verify_shape"] = verdict.schema_tag == "byok_audit.v1" and isinstance(
        verdict.errors, tuple
    )
    tampered = dict(good)
    tampered["receipt_sha256"] = "0" * 64
    out["verify_tamper_detected"] = sdk.verify_receipt(tampered).valid is False
    out["verify_malformed_fails_closed"] = sdk.verify_receipt({"not": "receipt"}).valid is False

    # ---- health ----------------------------------------------------------------
    h = sdk.health()
    out["health_booleans_only"] = all(isinstance(v, bool) for v in h.backends.values()) and set(
        h.backends
    ) == {"hosted_k3", "byok", "local_fx1"}
    out["health_no_values"] = all(not isinstance(v, str) for v in h.backends.values())
    out["health_commands_count"] = h.registered_commands == len(sdk.commands())
    out["health_version"] = h.version != ""

    # local_fx1 presence needs card'd checkpoint AND serve env
    envs = ["FX1_CHECKPOINT_DIR", "FX1_LOCAL_SERVE_URL", "FX1_LOCAL_SERVE_CMD"]
    saved = {k: os.environ.pop(k, None) for k in envs}
    try:
        with tempfile.TemporaryDirectory() as td2:
            os.environ["FX1_CHECKPOINT_DIR"] = td2
            out["health_localfx1_needs_card"] = sdk.health().backends["local_fx1"] is False
            (Path(td2) / "modelcard.json").write_text("{}")
            out["health_localfx1_needs_serve_env"] = sdk.health().backends["local_fx1"] is False
            os.environ["FX1_LOCAL_SERVE_URL"] = "http://127.0.0.1:9"
            out["health_localfx1_configured"] = sdk.health().backends["local_fx1"] is True
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ---- /v1/evals spec+run twins ------------------------------------------
    sdk_e = Fx1Harness(backend_resolver=lambda *a, **k: fake)
    spec = sdk_e.eval_spec_create("spec-a", suite="tooluse", seed=0, metadata={"lane": "audit"})
    out["evalspec_create"] = (
        spec["id"].startswith("eval_")
        and spec["object"] == "eval"
        and spec["name"] == "spec-a"
        and spec["data_source_config"]["item_schema"]["suite"] == "tooluse"
        and spec["metadata"] == {"lane": "audit"}
    )
    out["evalspec_bad_suite_rejected"] = (
        _raises(lambda: sdk_e.eval_spec_create("x", suite="nope")) == "ValidationError"
    )
    out["evalspec_list_get"] = (
        sdk_e.eval_specs(limit=5)[0]["id"] == spec["id"]
        and sdk_e.eval_spec_get(spec["id"])["id"] == spec["id"]
        and _raises(lambda: sdk_e.eval_spec_get("eval_nope")) == "KeyError"
    )
    out["evalspec_update"] = (
        sdk_e.eval_spec_update(spec["id"], name="renamed")["name"] == "renamed"
        and _raises(lambda: sdk_e.eval_spec_update(spec["id"])) == "ValueError"
    )
    run = sdk_e.eval_run_create(spec["id"], model="byok")
    out["evalrun_completed"] = (
        run["object"] == "eval.run"
        and run["id"].startswith("evalrun_")
        and run["eval_id"] == spec["id"]
        and run["model"] == "byok"
        and run["status"] == "completed"
        and "result_counts" in run
    )
    items = sdk_e.eval_run_items(spec["id"], run["id"])
    out["evalrun_items"] = (
        len(items) > 0
        and all(it["object"] == "eval.run.output_item" for it in items)
        and all(any("name" in r and "passed" in r for r in it["results"]) for it in items)
    )
    out["evalrun_list_get"] = (
        sdk_e.eval_runs(spec["id"])[0]["id"] == run["id"]
        and sdk_e.eval_run_get(spec["id"], run["id"])["id"] == run["id"]
    )
    out["evalrun_cross_spec_404"] = (
        _raises(lambda: sdk_e.eval_run_get("eval_nope", run["id"])) == "KeyError"
    )
    sdk_e.eval_run_delete(spec["id"], run["id"])
    sdk_e.eval_spec_delete(spec["id"])
    out["evalspec_run_delete"] = (
        _raises(lambda: sdk_e.eval_run_get(spec["id"], run["id"])) == "KeyError"
        and _raises(lambda: sdk_e.eval_spec_get(spec["id"])) == "KeyError"
        and _raises(lambda: sdk_e.eval_spec_delete(spec["id"])) == "KeyError"
    )

    # ---- /v1/uploads twin: chunked intent→parts→complete mints a file
    # in-process; md5 prechecked; terminal + bounds fail closed.
    import hashlib as _hashul  # noqa: PLC0415

    _ub = b'{"u":1}\n{"u":2}\n'
    _ucr = sdk.upload_create(bytes=len(_ub))
    _up1 = sdk.upload_part(_ucr["id"], _ub[:8])
    _up2 = sdk.upload_part(_ucr["id"], _ub[8:])
    _udone = sdk.upload_complete(
        _ucr["id"],
        [_up2["id"], _up1["id"]],
        md5=_hashul.md5(_ub[8:] + _ub[:8], usedforsecurity=False).hexdigest(),
    )
    out["upload_lifecycle"] = (
        _ucr["object"] == "upload"
        and _ucr["status"] == "pending"
        and _up1["object"] == "upload.part"
        and _udone["status"] == "completed"
        and _udone["file"]["bytes"] == len(_ub)
        and sdk.file_content(_udone["file"]["id"]) == _ub[8:] + _ub[:8]
        and "_content" not in sdk.file_card(_udone["file"]["id"])
    )
    out["upload_fail_closed"] = (
        _raises(lambda: sdk.upload_create(bytes=0)) == "UploadStoreError"
        and _raises(lambda: sdk.upload_part("upload_ghost", b"ab")) == "UploadStoreError"
        and _raises(lambda: sdk.upload_complete(sdk.upload_create(bytes=9)["id"], ["part_ghost"]))
        == "UploadStoreError"
        and sdk.upload_cancel(sdk.upload_create(bytes=4)["id"])["status"] == "cancelled"
    )

    # ---- harness bench: the perf gate times gated completes on either
    # leg and seals its record; prompts are digested, errors counted by
    # class, bounds fail closed before any token is spent.
    from fx1.harness_bench import run_bench as _run_bench  # noqa: PLC0415
    from fx1.sdk import CompletionResult as _CR  # noqa: PLC0415
    from fx1.serve.ops_receipt import bench_receipt as _bench_rcpt  # noqa: PLC0415
    from quant_fund.research.receipt_v2 import (  # noqa: PLC0415
        verify_receipt_payload as _vrp,
    )

    class _BenchStub:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Any, **kw: Any) -> Any:
            _ = messages
            self.calls += 1
            return _CR(
                backend=str(kw.get("backend")),
                model="stub-v0",
                content="ok",
                usage={"prompt_tokens": 5, "completion_tokens": 7},
            )

    class _BenchDead:
        def complete(self, messages: Any, **kw: Any) -> Any:
            raise ConnectionError("down")

    _bs = _BenchStub()
    _brec = _run_bench(
        _bs, n=4, concurrency=2, warmup=1, prompt="probe", backend="byok", mode="remote"
    )
    _bm = _brec["metrics"]
    out["bench_record_metrics"] = (
        _bm["measured_requests"] == 4
        and _bm["error_count"] == 0
        and _bm["prompt_tokens_total"] == 20
        and _bm["completion_tokens_total"] == 28
        and _bm["usage_reported"] == 4
        and _bm["models"] == ["stub-v0"]
        and _bm["backends"] == ["byok"]
        and _bs.calls == 5  # warmup + measured
        and _brec["mode"] == "remote"
        and "probe" not in repr(_brec)
        and bool(_brec["params"]["prompt_sha256"])
    )
    _bdead = _run_bench(_BenchDead(), n=3, concurrency=1, warmup=0)
    out["bench_error_histogram"] = (
        _bdead["metrics"]["error_count"] == 3
        and _bdead["metrics"]["errors"] == {"ConnectionError": 3}
        and abs(_bdead["metrics"]["error_rate"] - 1.0) < 1e-9
    )
    out["bench_bounds_fail_closed"] = (
        _raises(lambda: _run_bench(sdk, n=0)) == "ValueError"
        and _raises(lambda: _run_bench(sdk, concurrency=0)) == "ValueError"
        and _raises(lambda: _run_bench(sdk, warmup=-1)) == "ValueError"
        and _raises(lambda: _run_bench(sdk, max_tokens=0)) == "ValueError"
        and _raises(lambda: _run_bench(sdk, timeout_s=0.0)) == "ValueError"
        and _raises(lambda: _run_bench(sdk, prompt="  ")) == "ValueError"
        and _raises(lambda: _run_bench(sdk, mode="sideways")) == "ValueError"
    )
    out["bench_receipt_verifies"] = _vrp(_bench_rcpt(_brec))["valid"] is True

    # ---- usage accounting — the in-process twin of /harness/usage ---------
    # a usage-reporting backend + a dead link: totals, splits, filters, and
    # the truncation honesty fields all assert.
    class _UsageBackend(_FakeBackend):
        def complete(
            self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
        ) -> str:
            self.last_usage = {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
                "cached_tokens": 1,
            }
            return super().complete(messages, sampling=sampling)

    u_ok = _UsageBackend()

    class _DeadBackend:
        def complete(self, messages: Any, **kw: Any) -> str:
            raise RuntimeError("dead")

        def close(self) -> None:
            pass

    def _u_resolve(name: str, *a: Any, **k: Any) -> Any:
        if name == "byok":
            return u_ok
        return _DeadBackend()

    sdk_u = Fx1Harness(backend_resolver=_u_resolve)
    sdk_u.complete(
        [{"role": "user", "content": "hi"}],
        backend="byok",
        byok={"base_url": "http://u.test", "api_key": "k", "model": "m"},
    )
    sdk_u.complete(
        [{"role": "user", "content": "hi"}],
        backend="byok",
        byok={"base_url": "http://u.test", "api_key": "k", "model": "m"},
    )
    out["usage_failed_call_logged"] = (
        _raises(
            lambda: sdk_u.complete(
                [{"role": "user", "content": "hi"}],
                backend="hosted_k3",
            )
        )
        == "RuntimeError"
    )
    rep = sdk_u.usage()
    out["usage_totals"] = (
        rep.totals.requests == 3
        and rep.totals.ok == 2
        and rep.totals.errors == 1
        and rep.totals.prompt_tokens == 8
        and rep.totals.completion_tokens == 12
        and rep.totals.total_tokens == 20
        and rep.totals.other_usage == {"cached_tokens": 2}
        and rep.totals.usage_reported == 2
        and rep.records_dropped == 0
        and rep.by_backend["byok"].requests == 2
        and rep.by_backend["hosted_k3"].errors == 1
        and rep.by_model["fake-0"].requests == 2
    )
    out["usage_filters"] = (
        sdk_u.usage(backend="byok").totals.requests == 2
        and sdk_u.usage(model="fake-0").totals.requests == 2
        and sdk_u.usage(model="nope").totals.requests == 0
        and sdk_u.usage(since=time.time() + 60).records_seen == 0
        and sdk_u.usage(until=1.0).records_seen == 0
    )
    out["usage_bad_window_raises"] = (
        _raises(lambda: sdk_u.usage(since=2.0, until=1.0)) == "ValueError"
        and _raises(lambda: sdk_u.usage(backend="nope")) == "ValueError"
    )

    return out


def sdk_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under sdk_audit.v1."""
    r = sdk_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "sdk_audit",
        "schema": "sdk_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Fx1Harness is the in-process twin of the harness HTTP API: "
            "identical registry, identical backend set and error taxonomy, "
            "honesty gate enforced before results return, spawned engines "
            "always closed, receipt verification parity with the wire "
            "surface, and health that leaks booleans only."
            if ok
            else f"SDK AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
