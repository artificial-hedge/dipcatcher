"""api_audit — adversarial probes on the fx-1 harness HTTP surface.

The harness API (``fx1.serve.api``) is the network half of the harness
contract: anything reachable over the socket must be exactly what the
in-process ``Harness`` allows — no more. These probes pin that parity:

- *Registry parity* — every command in ``HARNESS_REGISTRY`` is listed and
  executable; unregistered names 404 over the wire exactly as ``Harness.get``
  raises in-process.
- *Containment parity* — ``config`` / ``extra_args=--config`` escapes hit
  the same ``configs/`` allowlist and 422; a config keyword on a non-local
  backend is refused.
- *Fail-closed shape* — wrong field types, unknown keys, empty message
  lists, oversized bodies, and bad methods all 4xx uniformly; nothing
  reaches a route handler malformed.
- *Completion gate* — completions run through ``cited_complete``, so a
  backend that emits a forbidden headline is cut off at the gate (502)
  rather than served; unconfigured BYOK/hosted creds surface as 503, a
  backend that doesn't exist as 422. ``local_fx1`` completes through an
  attached/spawned OpenAI-compatible engine: no ``FX1_LOCAL_SERVE_URL`` /
  ``FX1_LOCAL_SERVE_CMD`` → 503, a dead engine → 502, a live one → 200
  with the card's version as the model id; ``/health`` reports
  ``local_fx1`` ready only when a card'd checkpoint *and* engine config
  are both present.
- *Verification surface* — ``POST /receipts/verify`` verifies arbitrary
  payloads through the same ``verify_receipt_payload`` the CLI uses;
  malformed receipts fail closed, never crash.
- *Auth posture* — ``/health`` is the only public route; ``FX1_API_KEY``
  set → every other route requires ``X-API-Key`` (compare_digest); unset →
  non-loopback clients 403. Security headers on every response.

Sealed ``api_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["api_audit", "api_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_BYOK_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")
_LOCAL_ENVS = (
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
)


def _client(
    api_key: str | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    # An injected runner keeps the audit hermetic — probes never spawn real
    # lab processes, exactly like harness_audit's in-process probes.
    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, f"ran:{' '.join(argv)}", ""

    saved = {
        k: os.environ.get(k) for k in (_API_KEY_ENV, *_BYOK_ENVS, *_LOCAL_ENVS, "MOONSHOT_API_KEY")
    }
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(harness=Harness(runner=fake_runner))
        return TestClient(app, raise_server_exceptions=False), api_mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def api_audit() -> dict[str, Any]:
    out: dict[str, Any] = {}

    client, api_mod = _client()
    from fx1.harness import HARNESS_REGISTRY

    # --- registry parity ---------------------------------------------------
    resp = client.get("/harness/commands")
    names = {c["name"] for c in resp.json()["items"]}
    out["commands_all_listed"] = resp.status_code == 200 and names == {
        c.name for c in HARNESS_REGISTRY
    }
    resp = client.get("/harness/commands", params={"role": "verification"})
    out["commands_role_filter"] = resp.status_code == 200 and all(
        c["role"] == "verification" for c in resp.json()["items"]
    )
    out["commands_role_bad_422"] = (
        client.get("/harness/commands", params={"role": "bogus"}).status_code == 422
    )

    # --- execution surface ---------------------------------------------------
    resp = client.post("/harness/runs", json={"command": "doctor"})
    body = resp.json()
    out["run_executes"] = (
        resp.status_code == 200
        and body["ok"] is True
        and body["exit_code"] == 0
        and "doctor" in body["stdout"]
    )
    out["run_unknown_404"] = (
        client.post("/harness/runs", json={"command": "pwn"}).status_code == 404
    )
    out["run_extra_key_422"] = (
        client.post("/harness/runs", json={"command": "doctor", "evil": True}).status_code == 422
    )
    out["run_args_wrongtype_422"] = (
        client.post(
            "/harness/runs", json={"command": "doctor", "extra_args": "rm -rf /"}
        ).status_code
        == 422
    )
    out["run_method_shape"] = client.get("/harness/runs").status_code == 405
    out["run_timeout_pinned"] = resp.json()["timeout_s"] > 0 if resp.status_code == 200 else False

    # --- containment parity ------------------------------------------------
    out["config_escape_422"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "config": "/etc/passwd"},
        ).status_code
        == 422
    )
    out["extra_args_config_escape_422"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "extra_args": ["--config", "/etc/passwd"]},
        ).status_code
        == 422
    )
    out["extra_args_config_eq_escape_422"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "extra_args": ["--config=/etc/passwd"]},
        ).status_code
        == 422
    )

    # --- completion surface --------------------------------------------------
    for name in _BYOK_ENVS:
        os.environ.pop(name, None)
    os.environ.pop("MOONSHOT_API_KEY", None)
    out["complete_byok_unconfigured_503"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 503
    )
    out["complete_hosted_unconfigured_503"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 503
    )
    out["complete_unknown_backend_422"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "bogus",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 422
    )
    out["complete_empty_messages_422"] = (
        client.post("/harness/complete", json={"backend": "byok", "messages": []}).status_code
        == 422
    )
    out["complete_checkpoint_misplaced_422"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "ping"}],
                "checkpoint_dir": "./no-such-dir",
            },
        ).status_code
        == 422
    )
    out["complete_local_no_checkpoint_422"] = (
        client.post(
            "/harness/complete",
            json={
                "backend": "local_fx1",
                "messages": [{"role": "user", "content": "ping"}],
            },
        ).status_code
        == 422
    )

    # --- local weights surface: card'd checkpoint + engine attach ----------
    import json as _json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from fx1.modelcard import EvalDelta, ModelCard

    card = ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="aa..bb",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.6,
            general_pass_rate_base=0.5,
            general_pass_rate_candidate=0.5,
            honesty_gate_candidate=True,
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt"
        ckpt.mkdir()
        card.save(ckpt / "modelcard.json")

        out["local_unconfigured_503"] = (
            client.post(
                "/harness/complete",
                json={
                    "backend": "local_fx1",
                    "messages": [{"role": "user", "content": "ping"}],
                    "checkpoint_dir": str(ckpt),
                },
            ).status_code
            == 503
        )
        os.environ["FX1_CHECKPOINT_DIR"] = str(ckpt)
        try:
            out["health_local_needs_engine"] = (
                client.get("/health").json()["backends"]["local_fx1"] is False
            )
        finally:
            os.environ.pop("FX1_CHECKPOINT_DIR", None)

        class _Resp:
            def read(self) -> bytes:
                return _json.dumps({"choices": [{"message": {"content": "answer"}}]}).encode()

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        def fake_urlopen(req: Any, **kw: Any) -> _Resp:
            return _Resp()

        import urllib.request  # noqa: PLC0415

        os.environ["FX1_CHECKPOINT_DIR"] = str(ckpt)
        os.environ["FX1_LOCAL_SERVE_URL"] = "http://127.0.0.1:8011/v1"
        try:
            with patch.object(urllib.request, "urlopen", fake_urlopen):
                resp = client.post(
                    "/harness/complete",
                    json={
                        "backend": "local_fx1",
                        "messages": [{"role": "user", "content": "ping"}],
                        "checkpoint_dir": str(ckpt),
                    },
                )
            body = resp.json()
            out["local_attach_complete_200"] = (
                resp.status_code == 200
                and body["content"].startswith("answer")
                and body["model"] == "fx-1.v0.1"
            )
            out["health_local_configured"] = (
                client.get("/health").json()["backends"]["local_fx1"] is True
            )
        finally:
            os.environ.pop("FX1_CHECKPOINT_DIR", None)
            os.environ.pop("FX1_LOCAL_SERVE_URL", None)
        os.environ["FX1_LOCAL_SERVE_URL"] = "http://127.0.0.1:9/v1"
        try:
            out["local_engine_down_502"] = (
                client.post(
                    "/harness/complete",
                    json={
                        "backend": "local_fx1",
                        "messages": [{"role": "user", "content": "ping"}],
                        "checkpoint_dir": str(ckpt),
                    },
                ).status_code
                == 502
            )
        finally:
            os.environ.pop("FX1_LOCAL_SERVE_URL", None)

    # honesty gate fires over the wire: a backend emitting a forbidden
    # headline must not serve it.
    class _DirtyBackend:
        def complete(self, messages: list[dict[str, str]]) -> str:
            return "The strategy achieved a sharpe of 2.1 on the tape."

    # Injected resolver — the honesty gate must hold even when the model
    # behind the socket is adversarial.
    from fastapi.testclient import TestClient as _TC2

    dirty = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _DirtyBackend()))
    out["complete_honesty_gate_502"] = (
        dirty.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "report"}],
            },
        ).status_code
        == 502
    )

    # --- batch completions ----------------------------------------------------
    # One backend instance serves the whole batch; per-item verdicts.
    class _CleanBackend:
        def __init__(self) -> None:
            self._model = "fake-0"
            self.calls = 0
            self.closed = 0

        def complete(self, messages: list[dict[str, str]]) -> str:
            self.calls += 1
            return f"clean:{messages[-1]['content']}"

        def close(self) -> None:
            self.closed += 1

    clean = _CleanBackend()
    resolves = [0]

    def _resolve_once(*a: Any, **k: Any) -> _CleanBackend:
        resolves[0] += 1
        return clean

    batch_client = _TC2(api_mod.create_app(backend_resolver=_resolve_once))
    r = batch_client.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [[{"role": "user", "content": f"q{i}"}] for i in range(3)],
        },
    )
    bj = r.json()
    out["batch_complete_200"] = (
        r.status_code == 200 and len(bj["results"]) == 3 and all(i["ok"] for i in bj["results"])
    )
    out["batch_in_order"] = [i["content"] for i in bj["results"]] == [
        "clean:q0",
        "clean:q1",
        "clean:q2",
    ]
    out["batch_one_backend"] = resolves[0] == 1 and clean.calls == 3
    out["batch_closed_once"] = clean.closed == 1
    out["batch_empty_422"] = (
        batch_client.post(
            "/harness/complete/batch", json={"backend": "byok", "batch": []}
        ).status_code
        == 422
    )
    out["batch_local_no_checkpoint_422"] = (
        batch_client.post(
            "/harness/complete/batch",
            json={
                "backend": "local_fx1",
                "batch": [[{"role": "user", "content": "x"}]],
            },
        ).status_code
        == 422
    )
    out["batch_unknown_backend_422"] = (
        batch_client.post(
            "/harness/complete/batch",
            json={"backend": "bogus", "batch": [[{"role": "u", "content": "x"}]]},
        ).status_code
        == 422
    )

    # Per-slot verdicts: a refusal on one item doesn't lose the batch.
    class _PartialBackend(_CleanBackend):
        def complete(self, messages: list[dict[str, str]]) -> str:
            if messages[-1]["content"] == "bad":
                return "total Sharpe 4.2 on NAV"  # forbidden headline
            return super().complete(messages)

    partial_client = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _PartialBackend()))
    r2 = partial_client.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [
                [{"role": "user", "content": "good"}],
                [{"role": "user", "content": "bad"}],
                [{"role": "user", "content": "good2"}],
            ],
        },
    )
    rj = r2.json()["results"]
    out["batch_per_item_verdicts"] = (
        r2.status_code == 200
        and rj[0]["ok"]
        and rj[2]["ok"]
        and not rj[1]["ok"]
        and rj[1]["error_class"] == "honesty_refusal"
    )

    # --- SSE streaming ---------------------------------------------------------
    class _StreamBackend(_CleanBackend):
        def stream(self, messages: list[dict[str, str]]) -> Any:
            yield "tok-a"
            yield "tok-b"

    stream_client = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _StreamBackend()))
    rs = stream_client.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "hi"}],
            "receipt_hashes": ["a" * 64],
        },
    )
    body_text = rs.text
    frames = [ln for ln in body_text.split("\n\n") if ln.strip()]
    out["stream_200_sse"] = (
        rs.status_code == 200
        and rs.headers.get("content-type", "").startswith("text/event-stream")
        and len(frames) == 5  # 2 tokens + footer + final + [DONE]
    )
    payloads = [
        _json.loads(ln[len("data: ") :])
        for ln in frames
        if ln.startswith("data: ") and ln[len("data: ") :].strip() != "[DONE]"
    ]
    out["stream_token_order"] = [p["content"] for p in payloads if p.get("type") == "token"][
        :2
    ] == ["tok-a", "tok-b"]
    out["stream_final_envelope"] = payloads[-1].get("type") == "final" and payloads[-1].get(
        "receipt_hashes"
    ) == ["a" * 64]
    out["stream_done_terminates"] = frames[-1].strip() == "data: [DONE]"
    out["stream_footer_cited"] = any("Evidence:" in p.get("content", "") for p in payloads)

    class _DirtyStreamBackend(_DirtyBackend):
        def stream(self, messages: list[dict[str, str]]) -> Any:
            yield "total Sharpe 4.2 on NAV"

    dirty_stream = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _DirtyStreamBackend()))
    rd = dirty_stream.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    # Gate fires before any SSE frame is emitted — the refusal is a plain
    # JSON error, never a truncated event stream.
    out["stream_gate_502_json"] = (
        rd.status_code == 502
        and not rd.headers.get("content-type", "").startswith("text/event-stream")
        and "total Sharpe" not in rd.text
    )
    out["stream_local_no_checkpoint_422"] = (
        stream_client.post(
            "/harness/complete/stream",
            json={"backend": "local_fx1", "messages": [{"role": "u", "content": "x"}]},
        ).status_code
        == 422
    )
    out["stream_unsupported_501"] = (
        batch_client.post(  # _CleanBackend has no stream()
            "/harness/complete/stream",
            json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
        ).status_code
        == 501
    )

    # --- SSE keepalive (grace window → comment frames → in-band errors) ----
    class _SlowStreamBackend(_CleanBackend):
        def stream(self, messages: list[dict[str, str]]) -> Any:
            time.sleep(0.3)
            yield "slow-tok"

    ka_client = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _SlowStreamBackend(),
            sse_keepalive_s=0.05,
        )
    )
    rk = ka_client.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    ka_frames = [ln for ln in rk.text.split("\n\n") if ln.strip()]
    out["stream_keepalive_comments"] = (
        rk.status_code == 200
        and rk.headers.get("content-type", "").startswith("text/event-stream")
        and any(ln.strip() == ": keepalive" for ln in ka_frames)
        and '"type": "final"' in rk.text
        and ka_frames[-1].strip() == "data: [DONE]"
    )

    class _SlowFailBackend(_CleanBackend):
        def stream(self, messages: list[dict[str, str]]) -> Any:
            time.sleep(0.3)
            raise RuntimeError("engine died mid-generation")

    ka_fail = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _SlowFailBackend(),
            sse_keepalive_s=0.05,
        )
    )
    rf = ka_fail.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    err_frames = [
        _json.loads(ln[len("data: ") :])
        for ln in rf.text.split("\n\n")
        if ln.startswith("data: ") and '"type": "error"' in ln
    ]
    out["stream_keepalive_error_inband"] = (
        rf.status_code == 200
        and len(err_frames) == 1
        and err_frames[0]["status"] == 502
        and "engine died" in err_frames[0]["detail"]
        and rf.text.rstrip().endswith("data: [DONE]")
    )

    # Resolved inside the grace window → the synchronous contract holds:
    # refusals are still JSON errors, completions still plain SSE.
    ka_dirty = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _DirtyStreamBackend(),
            sse_keepalive_s=5.0,
        )
    )
    rg = ka_dirty.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    out["stream_keepalive_grace_json_error"] = (
        rg.status_code == 502
        and not rg.headers.get("content-type", "").startswith("text/event-stream")
        and "total Sharpe" not in rg.text
    )
    ka_fast = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _StreamBackend(),
            sse_keepalive_s=5.0,
        )
    )
    rfast = ka_fast.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    out["stream_keepalive_grace_sse"] = (
        rfast.status_code == 200
        and ": keepalive" not in rfast.text
        and rfast.text.rstrip().endswith("data: [DONE]")
    )

    # Disabled → fully synchronous; misconfig fails closed.
    off_client = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _DirtyStreamBackend(),
            sse_keepalive_s=0,
        )
    )
    out["stream_keepalive_off_json_error"] = (
        off_client.post(
            "/harness/complete/stream",
            json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
        ).status_code
        == 502
    )
    try:
        api_mod.create_app(sse_keepalive_s=-1.0)
        out["stream_keepalive_negative_rejected"] = False
    except ValueError:
        out["stream_keepalive_negative_rejected"] = True
    keep_env = os.environ.get("FX1_API_SSE_KEEPALIVE_S")
    os.environ["FX1_API_SSE_KEEPALIVE_S"] = "7.5"
    try:
        out["stream_keepalive_env_config"] = api_mod.create_app().state.sse_keepalive_s == 7.5
    finally:
        if keep_env is None:
            os.environ.pop("FX1_API_SSE_KEEPALIVE_S", None)
        else:
            os.environ["FX1_API_SSE_KEEPALIVE_S"] = keep_env

    # --- drain: one-way latch, gated routes refuse, ops routes stay up ----
    from fx1.harness import Harness as _Harness  # noqa: PLC0415

    drain_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    dclient = _TC2(drain_app)
    drain_app.state.metrics.acquire()  # pretend one request is in-flight
    # a stored key must still serve its replay while draining
    dclient.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "pre-drain-key"},
    )
    try:
        ready0 = dclient.get("/ready")
        out["ready_200"] = (
            ready0.status_code == 200
            and ready0.json()["ready"] is True
            and ready0.json()["inflight"] >= 1
        )
        d0 = dclient.post("/harness/drain")
        out["drain_response_shape"] = (
            d0.status_code == 200
            and d0.json()["draining"] is True
            and isinstance(d0.json()["inflight"], int)
        )
        out["drain_reports_inflight"] = d0.json()["inflight"] >= 1
        out["drain_complete_refused_503"] = (
            dclient.post(
                "/harness/complete",
                json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
            ).status_code
            == 503
        )
        out["drain_runs_refused_503"] = (
            dclient.post("/harness/runs", json={"command": "x"}).status_code == 503
        )
        out["drain_uncapped_routes_up"] = (
            dclient.get("/harness/commands").status_code == 200
            and dclient.post("/receipts/verify", json={"receipt": {"x": 1}}).status_code == 200
        )
        out["drain_health_reports"] = (
            dclient.get("/health").status_code == 200
            and dclient.get("/health").json()["draining"] is True
        )
        out["drain_metrics_reports"] = dclient.get("/metrics").json()["draining"] is True
        out["drain_idempotent"] = dclient.post("/harness/drain").json()["draining"] is True
        out["ready_under_drain_503"] = dclient.get("/ready").status_code == 503
        out["error_code_draining"] = (
            dclient.post("/harness/runs", json={"command": "x"}).json()["code"] == "draining"
        )
        replayed = dclient.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "pre-drain-key"},
        )
        out["drain_replay_served"] = (
            replayed.status_code == 200 and replayed.json()["replayed"] is True
        )
    finally:
        drain_app.state.metrics.release()

    # --- drain wait_s: server-side wait for the in-flight pool --------------
    import time as _time  # noqa: PLC0415

    def _slow_runner(a: list[str], t: float) -> tuple[int, str, str]:
        _time.sleep(0.4)
        return (0, "ran:" + " ".join(a), "")

    wait_app = api_mod.create_app(
        harness=_Harness(runner=_slow_runner),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    wclient = _TC2(wait_app)
    wjob = wclient.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    w0 = wclient.post("/harness/drain?wait_s=0.01")
    out["drain_wait_s_timeout_reports_not_drained"] = (
        w0.status_code == 200 and w0.json()["drained"] is False
    )
    w1 = wclient.post("/harness/drain?wait_s=5")
    out["drain_wait_s_blocks_until_empty"] = (
        w1.status_code == 200 and w1.json()["drained"] is True and w1.json()["inflight"] == 0
    )
    out["drain_wait_s_job_completes"] = (
        wclient.get(f"/harness/jobs/{wjob}").json()["status"] == "succeeded"
    )
    out["drain_wait_s_validated"] = wclient.post("/harness/drain?wait_s=-1").status_code == 422

    # --- idempotency keys: dedup retries of a submitted run ------------------
    r1 = client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "audit-key-1"},
    )
    r2 = client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "audit-key-1"},
    )
    out["idem_same_key_replays"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r1.json()["replayed"] is False
        and r2.json()["replayed"] is True
        and r2.json()["stdout"] == r1.json()["stdout"]
    )
    r3 = client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "audit-key-2"},
    )
    out["idem_distinct_keys_fresh"] = r3.json()["replayed"] is False
    out["idem_absent_ok"] = (
        client.post("/harness/runs", json={"command": "doctor"}).json()["replayed"] is False
    )
    out["idem_oversized_400"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "k" * 300},
        ).status_code
        == 400
    )
    tiny_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        idem_max=2,
    )
    tc = _TC2(tiny_app)
    for k_ in ("tk1", "tk2", "tk3"):
        tc.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": k_},
        )
    out["idem_bound_evicts_oldest"] = (
        tc.post(
            "/harness/runs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "tk1"},
        ).json()["replayed"]
        is False
    )
    try:
        api_mod.create_app(idem_max=0)
        out["idem_max_validated"] = False
    except ValueError:
        out["idem_max_validated"] = True

    # --- async jobs ------------------------------------------------------------
    jobs_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    jclient = _TC2(jobs_app)
    submit = jclient.post("/harness/jobs", json={"command": "doctor"})
    out["job_submit_202"] = submit.status_code == 202
    job_id = submit.json()["job_id"]
    out["job_id_shape"] = bool(job_id) and len(job_id) == 32
    job = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        job = jclient.get(f"/harness/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)
    out["job_completes_succeeded"] = job is not None and job["status"] == "succeeded"
    out["job_result_fields"] = (
        job is not None
        and job["result"] is not None
        and job["result"]["command"] == "doctor"
        and job["result"]["ok"] is True
        and job["finished_at"] is not None
    )
    out["job_unknown_submit_404"] = (
        jclient.post("/harness/jobs", json={"command": "nope-nope"}).status_code == 404
    )
    out["job_status_unknown_404"] = jclient.get("/harness/jobs/does-not-exist").status_code == 404
    replay = jclient.post(
        "/harness/jobs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "job-key-1"},
    )
    replay2 = jclient.post(
        "/harness/jobs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "job-key-1"},
    )
    out["job_idem_replay"] = (
        replay.status_code == 202
        and replay2.json()["job_id"] == replay.json()["job_id"]
        and replay2.json()["replayed"] is True
    )
    out["job_idem_conflict_409"] = (
        jclient.post(
            "/harness/jobs",
            json={"command": "doctor", "extra_args": ["--x"]},
            headers={"Idempotency-Key": "job-key-1"},
        ).status_code
        == 409
    )
    # inventory: newest-first listing with status filter + offset/limit paging
    lst = jclient.get("/harness/jobs").json()
    out["job_list_200"] = "jobs" in lst and isinstance(lst["total"], int) and lst["total"] >= 1
    out["job_list_status_filter"] = all(
        j["status"] == "succeeded"
        for j in jclient.get("/harness/jobs", params={"status": "succeeded"}).json()["jobs"]
    )
    page1 = jclient.get("/harness/jobs", params={"limit": 1}).json()
    page2 = jclient.get("/harness/jobs", params={"limit": 1, "offset": 1}).json()
    out["job_list_paging"] = (
        len(page1["jobs"]) == 1
        and page2["jobs"]
        and page1["jobs"][0]["job_id"] != page2["jobs"][0]["job_id"]
    )
    out["job_list_bad_status_422"] = (
        jclient.get("/harness/jobs", params={"status": "bogus"}).status_code == 422
    )
    # drain: new submissions refused; existing records still readable
    jclient.post("/harness/drain")
    out["job_drain_refuses_submit"] = (
        jclient.post("/harness/jobs", json={"command": "doctor"}).status_code == 503
    )
    out["job_status_under_drain"] = jclient.get(f"/harness/jobs/{job_id}").status_code == 200
    out["job_replay_under_drain"] = (
        jclient.post(
            "/harness/jobs",
            json={"command": "doctor"},
            headers={"Idempotency-Key": "job-key-1"},
        ).json()["replayed"]
        is True
    )
    # store bound: evicting the oldest job drops its idem mapping
    tiny_jobs = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        job_max=2,
    )
    tc3 = _TC2(tiny_jobs)
    ids = [tc3.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"] for _ in range(3)]
    out["job_bound_evicts"] = (
        tc3.get(f"/harness/jobs/{ids[0]}").status_code == 404
        and tc3.get(f"/harness/jobs/{ids[2]}").status_code == 200
    )
    try:
        api_mod.create_app(job_max=0)
        out["job_max_validated"] = False
    except ValueError:
        out["job_max_validated"] = True

    # cooperative cancel: occupy both executor workers without slots
    # (slots == workers, so jobs only stay queued when a worker is busy
    # without holding one) -> both jobs queue deterministically.
    qapp = api_mod.create_app(
        harness=_Harness(runner=_slow_runner),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=2,
    )
    qc = _TC2(qapp)
    qapp.state.jobs_executor.submit(lambda: time.sleep(2.5))
    qapp.state.jobs_executor.submit(lambda: time.sleep(2.5))
    j1 = qc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    j2 = qc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    out["job_queues_when_workers_busy"] = (
        qc.get(f"/harness/jobs/{j2}").json()["status"] == "queued"
        and qc.get(f"/harness/jobs/{j1}").json()["status"] == "queued"
    )
    cancelled = qc.delete(f"/harness/jobs/{j2}")
    out["job_cancel_queued_200"] = (
        cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    )
    # DELETE is idempotent: re-cancelling the cancelled job re-reads 200
    out["job_cancel_idempotent_200"] = qc.delete(f"/harness/jobs/{j2}").status_code == 200
    out["job_cancel_unknown_404"] = qc.delete("/harness/jobs/nope").status_code == 404
    out["job_cancelled_never_runs"] = (
        qc.get(f"/harness/jobs/{j2}").json()["result"] is None
        and qc.get(f"/harness/jobs/{j2}").json()["finished_at"] is not None
    )
    # once a worker frees, j1 dequeues; deleting a live or finished job is 409
    dl = time.monotonic() + 10.0
    while qc.get(f"/harness/jobs/{j1}").json()["status"] == "queued" and time.monotonic() < dl:
        time.sleep(0.05)
    out["job_cancel_active_409"] = qc.delete(f"/harness/jobs/{j1}").status_code == 409
    dl = time.monotonic() + 10.0
    while qc.get(f"/harness/jobs/{j1}").json()["status"] not in ("succeeded", "failed") and (
        time.monotonic() < dl
    ):
        time.sleep(0.05)
    out["job_cancel_succeeded_409"] = qc.delete(f"/harness/jobs/{j1}").status_code == 409
    # the cancelled job's slot frees once its executor task dequeues
    dl = time.monotonic() + 10.0
    while qc.get("/metrics").json()["inflight"] != 0 and time.monotonic() < dl:
        time.sleep(0.1)
    out["job_cancel_slot_recovered"] = qc.get("/metrics").json()["inflight"] == 0

    # --- receipt verification -------------------------------------------------
    from fx1.serve.byok_audit import byok_audit_bench

    good = byok_audit_bench()
    resp = client.post("/receipts/verify", json={"receipt": good})
    out["receipt_valid_payload"] = resp.status_code == 200 and resp.json()["valid"] is True
    tampered = dict(good)
    tampered["receipt_sha256"] = "0" * 64
    resp = client.post("/receipts/verify", json={"receipt": tampered})
    out["receipt_tamper_detected"] = resp.status_code == 200 and resp.json()["valid"] is False
    resp = client.post("/receipts/verify", json={"receipt": {"not": "a receipt"}})
    out["receipt_malformed_fails_closed"] = (
        resp.status_code == 200 and resp.json()["valid"] is False
    )
    out["receipt_non_object_422"] = (
        client.post("/receipts/verify", json={"receipt": [1, 2]}).status_code == 422
    )
    out["receipt_verify_never_5xx"] = client.post(
        "/receipts/verify",
        json={"receipt": {"nested": {"deep": [[{"x": None}]]}}},
    ).status_code in (200, 422)

    # --- auth + headers ---------------------------------------------------------
    health_resp = client.get("/health")
    out["health_public"] = health_resp.status_code == 200
    health = health_resp.json()
    out["health_no_secrets"] = all(isinstance(v, bool) for v in health["backends"].values()) and (
        os.environ.get("MOONSHOT_API_KEY", "\x00") not in health_resp.text
    )

    secured, _ = _client(api_key="k3y-material")
    out["auth_required_401"] = secured.get("/harness/commands").status_code == 401
    out["auth_wrong_key_401"] = (
        secured.get("/harness/commands", headers={"X-API-Key": "wrong"}).status_code == 401
    )
    out["auth_accepts_key"] = (
        secured.get("/harness/commands", headers={"X-API-Key": "k3y-material"}).status_code == 200
    )
    out["auth_health_still_public"] = secured.get("/health").status_code == 200

    out["loopback_served"] = client.get("/harness/commands").status_code == 200
    from fastapi.testclient import TestClient as _TC

    # A non-loopback client with no key configured is refused outright.
    remote_client = _TC(client.app, client=("203.0.113.7", 9))
    out["remote_refused_403"] = remote_client.get("/harness/commands").status_code == 403

    out["security_headers"] = all(
        h in client.get("/health").headers and h in client.get("/harness/commands").headers
        for h in ("x-content-type-options", "cache-control", "referrer-policy")
    )
    big = client.post(
        "/receipts/verify",
        content=b" " * ((1 << 20) + 1),
        headers={"content-type": "application/json"},
    )
    out["body_cap_413"] = big.status_code == 413

    # request tracing: X-Request-ID echoes when well-formed, mints otherwise
    echoed = client.get("/health", headers={"X-Request-ID": "trace-abc.123"})
    minted = client.get("/health")
    forged = client.get("/health", headers={"X-Request-ID": "bad\nid\x00inj"})
    out["request_id_echoed"] = echoed.headers.get("x-request-id") == "trace-abc.123"
    out["request_id_minted_when_absent"] = (
        minted.headers.get("x-request-id") is not None and len(minted.headers["x-request-id"]) == 32
    )
    out["request_id_malformed_replaced"] = (
        forged.headers.get("x-request-id") is not None
        and "\n" not in forged.headers["x-request-id"]
    )
    out["request_id_on_error_too"] = (
        secured.get("/harness/commands").headers.get("x-request-id") is not None
    )
    out["request_id_distinct"] = (
        client.get("/health").headers["x-request-id"]
        != client.get("/health").headers["x-request-id"]
    )

    # the response tail is single: body-cap / bad-length errors carry the
    # same security headers + request id as every other path
    bad_len = client.post(
        "/receipts/verify",
        content=b"{}",
        headers={"content-length": "abc"},
    )
    out["bad_content_length_400"] = bad_len.status_code == 400
    for label, err in (("cap_413", big), ("bad_len", bad_len)):
        hdrs = {k.lower(): v for k, v in err.headers.items()}
        out[f"error_tail_{label}"] = all(
            hdrs.get(k) is not None
            for k in (
                "x-request-id",
                "x-content-type-options",
                "cache-control",
                "referrer-policy",
            )
        )

    # --- error envelope: stable machine codes + API version negotiation ----
    v = client.get("/harness/version")
    out["version_route_200"] = (
        v.status_code == 200
        and v.json()["api_version"] == api_mod.API_VERSION
        and bool(v.json()["fx1_version"])
    )
    out["api_version_header_on_every_response"] = (
        client.get("/health").headers.get("x-fx1-api-version") == api_mod.API_VERSION
        and big.headers.get("x-fx1-api-version") == api_mod.API_VERSION
        and bad_len.headers.get("x-fx1-api-version") == api_mod.API_VERSION
    )
    out["error_code_not_found"] = (
        client.post("/harness/runs", json={"command": "pwn"}).json()["code"] == "not_found"
    )
    out["error_code_validation"] = (
        client.post("/harness/runs", json={"command": 1}).json()["code"] == "validation"
    )
    out["error_code_unauthorized"] = (
        secured.get("/harness/commands").json()["code"] == "unauthorized"
    )
    out["error_code_forbidden"] = (
        remote_client.get("/harness/commands").json()["code"] == "forbidden"
    )
    out["error_code_too_large"] = big.json()["code"] == "too_large"
    out["error_code_bad_request"] = bad_len.json()["code"] == "bad_request"
    client.post(
        "/harness/runs",
        json={"command": "doctor"},
        headers={"Idempotency-Key": "ec-conflict"},
    )
    out["error_code_conflict"] = (
        client.post(
            "/harness/runs",
            json={"command": "doctor", "extra_args": ["--x"]},
            headers={"Idempotency-Key": "ec-conflict"},
        ).json()["code"]
        == "conflict"
    )
    out["error_code_honesty_gate"] = rd.json()["code"] == "honesty_gate"
    out["stream_error_code_inband"] = err_frames[0]["code"] == "backend_failure"
    cap_app = api_mod.create_app(max_inflight=1)
    cap_client = _TC2(cap_app)
    cap_app.state.inflight_slots.acquire()
    try:
        capped = cap_client.post("/harness/runs", json={"command": "doctor"})
        out["error_code_over_capacity"] = (
            capped.status_code == 503 and capped.json()["code"] == "over_capacity"
        )
    finally:
        cap_app.state.inflight_slots.release()

    # one structured access line per request, keyed by the request id
    import logging  # noqa: PLC0415

    from fx1.serve.api import logger as api_logger  # noqa: PLC0415

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.lines: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.lines.append(record.getMessage())

    cap = _Capture()
    api_logger.addHandler(cap)
    try:
        client.get("/health", headers={"X-Request-ID": "rid-probe-1"})
    finally:
        api_logger.removeHandler(cap)
    line = next((ln for ln in cap.lines if "rid-probe-1" in ln), "")
    out["access_log_emitted"] = (
        "method=GET" in line
        and "path=/health" in line
        and "status=200" in line
        and "rid=rid-probe-1" in line
        and "elapsed_ms=" in line
    )

    # /metrics: ops snapshot — counters from the requests above (this very
    # request is recorded too, so totals are strictly increasing)
    m_before = client.get("/metrics")
    out["metrics_200"] = m_before.status_code == 200
    m = m_before.json()
    out["metrics_shape"] = all(
        k in m
        for k in (
            "uptime_s",
            "requests_total",
            "errors_total",
            "by_status",
            "inflight",
            "inflight_watermark",
            "max_inflight",
        )
    )
    out["metrics_counts_requests"] = m["requests_total"] >= 10 and "200" in m["by_status"]
    out["metrics_counts_errors"] = (
        m["errors_total"] >= 2 and "413" in m["by_status"] and "400" in m["by_status"]
    )
    out["metrics_inflight_idle"] = m["inflight"] == 0 and m["inflight_watermark"] >= 1
    out["metrics_config"] = m["max_inflight"] == 16 and m["uptime_s"] >= 0.0
    m2 = client.get("/metrics").json()
    out["metrics_monotone"] = m2["requests_total"] > m["requests_total"]
    out["metrics_secured_401"] = secured.get("/metrics").status_code == 401

    return out


def api_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under api_audit.v1."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = api_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "api_audit",
        "schema": "api_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Harness API holds: registry parity with the in-process Harness, "
            "configs/ containment over the wire, uniform 4xx fail-closed shape, "
            "the honesty gate fires on model output (502), credential-less "
            "backends 503, arbitrary receipts verify through the posted "
            "payload, and auth is X-API-Key or loopback-only."
            if ok
            else f"HARNESS API AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
