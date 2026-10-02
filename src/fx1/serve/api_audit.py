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

    # --- complete idempotency: retried submits must not re-bill the model ---
    from fastapi.testclient import TestClient as _IdemTC  # noqa: PLC0415

    class _IdemBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, str]]) -> str:
            self.calls += 1
            return f"clean:{messages[-1]['content']}"

    _idem_backend = _IdemBackend()
    idem_resolves = {"n": 0}

    def _counting_resolver(*a: Any, **k: Any) -> Any:
        idem_resolves["n"] += 1
        return _idem_backend

    idem_api = api_mod.create_app(backend_resolver=_counting_resolver)
    ic = _IdemTC(idem_api)
    cbody = {"backend": "byok", "messages": [{"role": "user", "content": "ping"}]}
    c1 = ic.post("/harness/complete", json=cbody, headers={"Idempotency-Key": "ck1"})
    c2 = ic.post("/harness/complete", json=cbody, headers={"Idempotency-Key": "ck1"})
    out["complete_idem_replay"] = (
        idem_resolves["n"] == 1
        and _idem_backend.calls == 1
        and c2.json()["replayed"] is True
        and c2.json()["content"] == c1.json()["content"]
    )
    out["complete_idem_conflict_409"] = (
        ic.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "other"}]},
            headers={"Idempotency-Key": "ck1"},
        ).status_code
        == 409
    )
    before = idem_resolves["n"]
    ic.post("/harness/complete", json=cbody)
    ic.post("/harness/complete", json=cbody)
    out["complete_no_key_reexecutes"] = idem_resolves["n"] == before + 2
    bbody = {"backend": "byok", "batch": [[{"role": "user", "content": "p"}]]}
    ic.post("/harness/complete/batch", json=bbody, headers={"Idempotency-Key": "bk1"})
    cb2 = ic.post("/harness/complete/batch", json=bbody, headers={"Idempotency-Key": "bk1"}).json()
    out["complete_batch_idem_replay"] = cb2["replayed"] is True
    out["complete_reports_latency_ms"] = (
        isinstance(c1.json()["latency_ms"], (int, float)) and c1.json()["latency_ms"] >= 0
    )
    out["complete_replay_reuses_latency_ms"] = c2.json()["latency_ms"] == c1.json()["latency_ms"]

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
    out["batch_item_latency_ms"] = all(
        isinstance(i["latency_ms"], (int, float)) and i["latency_ms"] >= 0 for i in bj["results"]
    )
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

    # no receipt store -> the synthetic citation below is advisory, not 422.
    stream_client = _TC2(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _StreamBackend(),
            receipts_dir="/nonexistent-stream-store",
        )
    )
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
    out["stream_final_reports_latency_ms"] = (
        isinstance(payloads[-1].get("latency_ms"), (int, float)) and payloads[-1]["latency_ms"] >= 0
    )

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
        out["drain_gate_check_up"] = (
            dclient.post("/harness/gate/check", json={"text": "ok"}).status_code == 200
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

    # --- webhooks: terminal-state job callbacks --------------------------------
    # A submitted job's record is POSTed to its callback_url on every
    # terminal transition (succeeded/failed/cancelled); delivery is
    # best-effort — a dead or erroring endpoint is recorded on the job,
    # never raised into the worker.
    import json as _json  # noqa: PLC0415
    import socket as _socket  # noqa: PLC0415
    import threading as _threading  # noqa: PLC0415
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    _cb_hits: list[dict[str, Any]] = []
    _cb_raw: list[bytes] = []
    _cb_hdrs: list[dict[str, str]] = []
    _cb_path_n: dict[str, int] = {}

    class _JobHook(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server handler name
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            _cb_raw.append(raw)
            _cb_hdrs.append(dict(self.headers.items()))
            _cb_hits.append(_json.loads(raw))
            _cb_path_n[self.path] = _cb_path_n.get(self.path, 0) + 1
            if self.path == "/fail":
                self.send_response(500)
            elif self.path == "/flaky":
                self.send_response(500 if _cb_path_n[self.path] < 3 else 200)
            elif self.path == "/reject":
                self.send_response(404)
            else:
                self.send_response(200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    cb_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=2,
    )
    cbc = _TC2(cb_app)
    cb_srv = ThreadingHTTPServer(("127.0.0.1", 0), _JobHook)
    cb_thread = _threading.Thread(target=cb_srv.serve_forever, daemon=True)
    cb_thread.start()
    cb_url = f"http://127.0.0.1:{cb_srv.server_address[1]}/hook"
    _dead_sock = _socket.socket()
    _dead_sock.bind(("127.0.0.1", 0))
    _dead_port = _dead_sock.getsockname()[1]
    _dead_sock.close()
    try:
        ok_job = cbc.post("/harness/jobs", json={"command": "doctor", "callback_url": cb_url})
        jid_cb = ok_job.json()["job_id"]
        deadline = time.monotonic() + 10.0
        st_cb: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_cb = cbc.get(f"/harness/jobs/{jid_cb}").json()
            if st_cb["status"] == "succeeded" and st_cb.get("callback_status"):
                break
            time.sleep(0.05)
        out["callback_delivered_on_success"] = (
            ok_job.status_code == 202
            and ok_job.headers.get("location") == f"/harness/jobs/{jid_cb}"
            and st_cb.get("callback_status") == "delivered"
            and st_cb.get("callback_url") == cb_url
            and len(_cb_hits) == 1
            and _cb_hits[0]["job_id"] == jid_cb
            and _cb_hits[0]["status"] == "succeeded"
            and _cb_hits[0]["result"]["ok"] is True
        )
        # endpoint returning 5xx -> recorded as failed, job unaffected
        err_job = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_url": cb_url.replace("/hook", "/fail")},
        )
        jid_err = err_job.json()["job_id"]
        deadline = time.monotonic() + 10.0
        st_err: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_err = cbc.get(f"/harness/jobs/{jid_err}").json()
            if st_err.get("callback_status") and st_err.get("callback_attempts") == 3:
                break
            time.sleep(0.05)
        out["callback_http_error_recorded"] = (
            st_err["status"] == "succeeded"
            and st_err.get("callback_status") == "failed"
            and "500" in (st_err.get("callback_error") or "")
        )
        out["callback_5xx_retried_3x"] = (
            _cb_path_n.get("/fail") == 3 and st_err.get("callback_attempts") == 3
        )

        # transient 5xx sequence eventually delivers; 4xx never retries.
        # Delivery retries hold a worker briefly — a submit can race a
        # saturated inflight cap, so re-submit until a slot frees.
        def _submit_cb(url: str) -> str:
            end = time.monotonic() + 10.0
            while True:
                r = cbc.post(
                    "/harness/jobs",
                    json={"command": "doctor", "callback_url": url},
                )
                if r.status_code == 202:
                    return str(r.json()["job_id"])
                if r.status_code != 503 or time.monotonic() > end:
                    raise AssertionError(f"callback submit failed: {r.status_code} {r.text}")
                time.sleep(0.1)

        jid_fk = _submit_cb(cb_url.replace("/hook", "/flaky"))
        deadline = time.monotonic() + 10.0
        st_fk: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_fk = cbc.get(f"/harness/jobs/{jid_fk}").json()
            if (
                st_fk["status"] == "succeeded"
                and st_fk.get("callback_status")
                and st_fk.get("callback_attempts") == 3
            ):
                break
            time.sleep(0.05)
        out["callback_flaky_delivers"] = (
            st_fk.get("callback_status") == "delivered"
            and st_fk.get("callback_attempts") == 3
            and _cb_path_n.get("/flaky") == 3
        )
        jid_rj = _submit_cb(cb_url.replace("/hook", "/reject"))
        deadline = time.monotonic() + 10.0
        st_rj: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_rj = cbc.get(f"/harness/jobs/{jid_rj}").json()
            if st_rj["status"] == "succeeded" and st_rj.get("callback_status"):
                break
            time.sleep(0.05)
        out["callback_4xx_never_retried"] = (
            st_rj.get("callback_status") == "failed"
            and st_rj.get("callback_attempts") == 1
            and _cb_path_n.get("/reject") == 1
        )
        # dead endpoint -> connection error recorded, job unaffected
        jid_dead = _submit_cb(f"http://127.0.0.1:{_dead_port}/hook")
        deadline = time.monotonic() + 15.0
        st_dead: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_dead = cbc.get(f"/harness/jobs/{jid_dead}").json()
            if (
                st_dead["status"] == "succeeded"
                and st_dead.get("callback_status")
                and st_dead.get("callback_attempts") == 3
            ):
                break
            time.sleep(0.05)
        out["callback_dead_endpoint_recorded"] = (
            st_dead["status"] == "succeeded"
            and st_dead.get("callback_status") == "failed"
            and bool(st_dead.get("callback_error"))
        )
        # cancelling a queued job is a terminal transition — it fires too
        cb_app.state.jobs_executor.submit(lambda: time.sleep(3.0))
        cb_app.state.jobs_executor.submit(lambda: time.sleep(3.0))
        qid = _submit_cb(cb_url)
        cxl = cbc.delete(f"/harness/jobs/{qid}")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and (
            len(_cb_hits) < 3 or _cb_hits[-1].get("job_id") != qid
        ):
            time.sleep(0.05)
        out["callback_fires_on_cancel"] = (
            cxl.status_code == 200
            and cxl.json()["status"] == "cancelled"
            and cxl.json().get("callback_status") == "delivered"
            and _cb_hits[-1]["job_id"] == qid
            and _cb_hits[-1]["status"] == "cancelled"
        )
        dl = time.monotonic() + 10.0
        while cbc.get("/metrics").json()["inflight"] != 0 and time.monotonic() < dl:
            time.sleep(0.1)
        bad = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_url": "ftp://x/hook"},
        )
        nohost = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_url": "http:///hook"},
        )
        out["callback_bad_url_422"] = bad.status_code == 422 and nohost.status_code == 422
        # unsigned deliveries carry no signature headers
        out["callback_unsigned_no_sig"] = (
            "X-Fx1-Webhook-Signature" not in _cb_hdrs[0]
            and "X-Fx1-Webhook-Timestamp" not in _cb_hdrs[0]
        )
        # callback_secret HMAC-signs the delivery; receivers verify the raw
        # body — the job record itself never echoes the secret.
        from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

        sig_job = cbc.post(
            "/harness/jobs",
            json={
                "command": "doctor",
                "callback_url": cb_url,
                "callback_secret": "whsec-test",
            },
        )
        jid_sig = sig_job.json()["job_id"]
        deadline = time.monotonic() + 10.0
        st_sig: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st_sig = cbc.get(f"/harness/jobs/{jid_sig}").json()
            if st_sig["status"] == "succeeded" and st_sig.get("callback_status"):
                break
            time.sleep(0.05)
        sig_hdrs = _cb_hdrs[-1]
        out["callback_signed_verifies"] = (
            st_sig.get("callback_status") == "delivered"
            and sig_hdrs.get("X-Fx1-Webhook-Signature", "").startswith("sha256=")
            and verify_webhook(
                "whsec-test",
                sig_hdrs.get("X-Fx1-Webhook-Timestamp"),
                sig_hdrs.get("X-Fx1-Webhook-Signature"),
                _cb_raw[-1],
            )
        )
        out["callback_secret_not_echoed"] = (
            "callback_secret" not in st_sig
            and "whsec-test" not in cbc.get(f"/harness/jobs/{jid_sig}").text
            and b"whsec-test" not in _cb_raw[-1]
        )
        good_sig = sig_hdrs.get("X-Fx1-Webhook-Signature")
        good_ts = sig_hdrs.get("X-Fx1-Webhook-Timestamp")
        out["webhook_verify_tampered_body"] = not verify_webhook(
            "whsec-test", good_ts, good_sig, _cb_raw[-1] + b" "
        )
        out["webhook_verify_wrong_secret"] = not verify_webhook(
            "whsec-other", good_ts, good_sig, _cb_raw[-1]
        )
        out["webhook_verify_stale_ts"] = not verify_webhook(
            "whsec-test", "1", good_sig, _cb_raw[-1]
        )
        out["webhook_verify_malformed"] = not verify_webhook(
            "whsec-test", good_ts, "nonsense", _cb_raw[-1]
        ) and not verify_webhook("whsec-test", good_ts, None, _cb_raw[-1])
        sec_no_url = cbc.post(
            "/harness/jobs",
            json={"command": "doctor", "callback_secret": "whsec-test"},
        )
        out["callback_secret_requires_url_422"] = sec_no_url.status_code == 422
    finally:
        cb_srv.shutdown()
        cb_srv.server_close()

    # --- rate limiting: per-client-host token bucket ----------------------------
    out["rate_limit_default_off"] = all(client.get("/health").status_code == 200 for _ in range(8))
    try:
        api_mod.create_app(rate_limit_rps=-1)
        out["rate_limit_negative_rejected"] = False
    except ValueError:
        out["rate_limit_negative_rejected"] = True
    limited = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=5.0,
    )
    lc = _TC2(limited)
    hits = [lc.get("/harness/version").status_code for _ in range(7)]
    denied = lc.get("/harness/version")
    out["rate_limit_429"] = hits[:5] == [200] * 5 and 429 in hits[5:] + [denied.status_code]
    out["rate_limit_envelope"] = (
        denied.status_code == 429
        and denied.json()["code"] == "too_many_requests"
        and denied.headers.get("retry-after") is not None
        and int(denied.headers["retry-after"]) >= 1
    )
    time.sleep(0.3)
    out["rate_limit_recovers"] = lc.get("/harness/version").status_code == 200
    # X-RateLimit-* headers on every response while the limiter is active
    limited3 = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=10.0,
    )
    lc3 = _TC2(limited3)
    spec_limited = lc3.get("/openapi.json").json()
    first = lc3.get("/harness/version")
    second = lc3.get("/harness/version")
    out["rate_limit_headers_on_success"] = (
        first.status_code == 200
        and first.headers.get("x-ratelimit-limit") == "10"
        and first.headers.get("x-ratelimit-remaining") == "8"
        and int(first.headers["x-ratelimit-reset"]) >= 0
        and int(second.headers["x-ratelimit-remaining"]) == 7
    )
    while lc3.get("/harness/version").status_code == 200:
        pass
    denied3 = lc3.get("/harness/version")
    out["rate_limit_headers_on_429"] = (
        denied3.status_code == 429
        and denied3.headers.get("x-ratelimit-limit") == "10"
        and denied3.headers.get("x-ratelimit-remaining") == "0"
        and int(denied3.headers["x-ratelimit-reset"]) >= 1
    )
    out["rate_limit_headers_absent_when_off"] = (
        "x-ratelimit-limit" not in client.get("/health").headers
    )
    # public paths are exempt: a health probe must not consume the budget —
    # with a 1 rps bucket, /version takes the only token, /health still 200s
    # (no bucket touch, no limiter headers), and the next /version 429s.
    lim_pub = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=1.0,
    )
    lp = _TC2(lim_pub)
    lp.get("/harness/version")
    h_exempt = lp.get("/health")
    v_after = lp.get("/harness/version")
    out["rate_limit_public_path_exempt"] = (
        h_exempt.status_code == 200
        and "x-ratelimit-limit" not in h_exempt.headers
        and v_after.status_code == 429
    )
    # the spec declares the headers the middleware sets — generated clients
    # see them typed instead of having to know
    spec_main = client.get("/openapi.json").json()
    common = {
        "X-Request-ID",
        "X-Fx1-Api-Version",
        "X-Content-Type-Options",
        "Cache-Control",
        "Referrer-Policy",
    }
    spec_ops = [
        op
        for item in spec_main.get("paths", {}).values()
        for op in item.values()
        if isinstance(op, dict)
    ]
    out["openapi_declares_common_headers"] = bool(spec_ops) and all(
        common <= set(resp.get("headers", {}))
        for op in spec_ops
        for resp in op.get("responses", {}).values()
    )
    submit_op = spec_main["paths"]["/harness/jobs"]["post"]
    out["openapi_declares_location_202"] = "Location" in submit_op["responses"]["202"].get(
        "headers", {}
    )
    out["openapi_declares_retry_after_503"] = all(
        "Retry-After" in resp.get("headers", {})
        for op in spec_ops
        for code, resp in op.get("responses", {}).items()
        if code in ("429", "503")
    )
    out["openapi_ratelimit_absent_when_off"] = all(
        "X-RateLimit-Limit" not in resp.get("headers", {})
        for op in spec_ops
        for resp in op.get("responses", {}).values()
    )
    out["openapi_ratelimit_present_when_on"] = all(
        {"X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"}
        <= set(resp.get("headers", {}))
        for item in spec_limited.get("paths", {}).values()
        for op in item.values()
        if isinstance(op, dict)
        for resp in op.get("responses", {}).values()
    )
    # CORS: closed by default; explicit origins only; preflight handled
    # before the auth layer (preflights carry no credentials)
    pre_off = client.options(
        "/health",
        headers={
            "Origin": "https://fx1.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    out["cors_off_by_default"] = "access-control-allow-origin" not in pre_off.headers
    cors_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        cors_origins="https://fx1.example.com, https://ops.internal:8443",
    )
    cc = _TC2(cors_app)
    pre_on = cc.options(
        "/health",
        headers={
            "Origin": "https://fx1.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    out["cors_preflight_ok"] = (
        pre_on.status_code == 200
        and pre_on.headers.get("access-control-allow-origin") == "https://fx1.example.com"
        and "POST" in pre_on.headers.get("access-control-allow-methods", "")
    )
    actual_resp = cc.get("/health", headers={"Origin": "https://fx1.example.com"})
    out["cors_exposes_stamped_headers"] = (
        "x-fx1-api-version" in actual_resp.headers.get("access-control-expose-headers", "").lower()
    )
    out["cors_wrong_origin_refused"] = (
        "access-control-allow-origin"
        not in cc.options(
            "/health",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        ).headers
    )
    out["cors_actual_response_marked"] = (
        actual_resp.headers.get("access-control-allow-origin") == "https://fx1.example.com"
    )
    try:
        api_mod.create_app(cors_origins="*")
        out["cors_wildcard_refused"] = False
    except ValueError:
        out["cors_wildcard_refused"] = True
    out["capabilities_reports_cors"] = (
        client.get("/harness/capabilities").json()["features"]["cors"] is False
        and cc.get("/harness/capabilities").json()["features"]["cors"] is True
    )

    # Backend circuit breaker: consecutive call faults open the circuit —
    # later calls fast-fail 503 + Retry-After without touching the backend;
    # a half-open probe admits after cooldown and closes on success.
    class _FlakyBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, str]]) -> str:
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("backend exploded")
            return "ok"

    _flaky = _FlakyBackend()
    brk_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda name, *a, **k: _flaky if name == "byok" else _CleanBackend(),
        breaker_threshold=2,
        breaker_cooldown_s=60.0,
    )
    bc = _TC2(brk_app)
    cp_body = {"backend": "byok", "messages": [{"role": "user", "content": "x"}]}
    r_fail1 = bc.post("/harness/complete", json=cp_body)
    r_fail2 = bc.post("/harness/complete", json=cp_body)
    r_open = bc.post("/harness/complete", json=cp_body)
    out["breaker_opens_at_threshold"] = (
        r_fail1.status_code == 502
        and r_fail2.status_code == 502
        and r_open.status_code == 503
        and r_open.json()["code"] == "backend_unavailable"
        and "circuit open" in r_open.json()["detail"]
        and "retry-after" in {k.lower() for k in r_open.headers}
        and _flaky.calls == 2  # the open-circuit call never reached the backend
    )
    st = bc.get("/harness/backends").json()
    out["backends_route_reports_state"] = (
        set(st) == {"hosted_k3", "byok", "local_fx1"}
        and st["byok"]["circuit_open"] is True
        and st["byok"]["consecutive_failures"] >= 2
        and st["hosted_k3"]["circuit_open"] is False
    )
    r_other = bc.post(
        "/harness/complete",
        json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "x"}]},
    )
    out["breaker_isolated_per_backend"] = r_other.status_code == 200
    caps_brk = bc.get("/harness/capabilities").json()
    out["capabilities_reports_breaker"] = (
        caps_brk["features"]["breaker"] is True
        and caps_brk["limits"]["breaker_threshold"] == 2.0
        and caps_brk["limits"]["breaker_cooldown_s"] == 60.0
    )
    brk = api_mod._BackendBreaker(2, 0.05)
    brk.report("x", False)
    brk.report("x", False)
    opened = brk.check("x") > 0
    time.sleep(0.06)
    probe1 = brk.check("x")  # admits the single half-open probe
    probe2 = brk.check("x")  # a second concurrent caller fast-fails
    brk.report("x", True)  # probe succeeded → closed
    closed = brk.check("x") == 0 and brk.state("x")[0] is False
    out["breaker_half_open_recovers"] = opened and probe1 == 0.0 and probe2 > 0 and closed
    off_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        breaker_threshold=0,
    )
    oc = _TC2(off_app)
    out["breaker_disabled_by_zero"] = (
        oc.get("/harness/capabilities").json()["features"]["breaker"] is False
        and oc.get("/harness/backends").json()["byok"]["circuit_open"] is False
    )
    # honesty-gate refusals are model output, not backend health — they
    # must never trip the circuit.
    dirty_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _DirtyBackend(),
        breaker_threshold=2,
        breaker_cooldown_s=60.0,
    )
    dc = _TC2(dirty_app)
    gate_fails = [
        dc.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "x"}]},
        ).status_code
        for _ in range(3)
    ]
    out["breaker_ignores_honesty_gate"] = (
        gate_fails == [502, 502, 502]
        and dc.get("/harness/backends").json()["byok"]["circuit_open"] is False
    )

    # Per-request BYOK: the caller's {base_url, api_key, model} rides the
    # request body and reaches the backend factory; overrides are validated
    # (422, never breaker-counted), gated by a server flag, and breaker-
    # isolated per endpoint so one tenant's dead endpoint can't fast-fail
    # another's.
    _cap: list[dict[str, Any]] = []
    _flaky2 = _FlakyBackend()
    _flaky2.calls = 0

    def _cap_resolver(name: str, cp: str | None = None, **kw: Any) -> Any:
        _cap.append(kw)
        if name == "byok" and "dead" in kw.get("base_url", ""):
            return _flaky2
        return _CleanBackend()

    bapp = _TC2(api_mod.create_app(backend_resolver=_cap_resolver))
    ovr = {"base_url": "https://llm.example.com/v1", "api_key": "sk-live-x", "model": "m1"}
    ok = bapp.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_reaches_backend"] = ok.status_code == 200 and _cap[-1] == ovr
    wrong_be = bapp.post(
        "/harness/complete",
        json={
            "backend": "hosted_k3",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_non_byok_422"] = wrong_be.status_code == 422
    out["byok_override_bad_url_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "byok": {**ovr, "base_url": "ftp://x"},
            },
        ).status_code
        == 422
    )
    out["byok_override_partial_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "byok": {"base_url": "https://e.com", "api_key": "k"},
            },
        ).status_code
        == 422
    )
    disabled = _TC2(api_mod.create_app(backend_resolver=_cap_resolver, byok_override=False))
    dresp = disabled.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_disabled_422"] = (
        dresp.status_code == 422 and dresp.json().get("code") == "byok_override_disabled"
    )
    caps_byok = bapp.get("/harness/capabilities").json()["features"]
    out["capabilities_reports_byok_override"] = (
        caps_byok["byok_override"] is True
        and _TC2(api_mod.create_app(backend_resolver=_cap_resolver, byok_override=False))
        .get("/harness/capabilities")
        .json()["features"]["byok_override"]
        is False
    )
    # Per-request backend deadline — flows into resolver kwargs on every
    # surface; out-of-range values are model-level 422s.
    _cap.clear()
    t_ok = bapp.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
            "timeout_s": 2.5,
        },
    )
    out["timeout_s_reaches_backend"] = t_ok.status_code == 200 and _cap[-1].get("timeout_s") == 2.5
    out["timeout_s_nonpositive_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "timeout_s": 0,
            },
        ).status_code
        == 422
    )
    out["timeout_s_over_cap_422"] = (
        bapp.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": [{"role": "user", "content": "x"}],
                "timeout_s": 99999,
            },
        ).status_code
        == 422
    )
    _cap.clear()
    bapp.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [[{"role": "user", "content": "x"}]],
            "byok": ovr,
            "timeout_s": 7.0,
        },
    )
    out["timeout_s_batch_reaches_backend"] = _cap[-1].get("timeout_s") == 7.0
    _cap.clear()
    bapp.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
            "timeout_s": 3.0,
        },
    )
    # _CleanBackend lacks stream() — the resolver still saw the deadline
    # before the 501.
    out["timeout_s_stream_reaches_backend"] = _cap[-1].get("timeout_s") == 3.0
    # The hosted backend honors the knob it previously hardcoded.
    import fx1.serve.backends as be_mod  # noqa: PLC0415

    out["timeout_s_hosted_backend"] = (
        be_mod.HostedK3Backend(api_key="k", timeout_s=9.0)._timeout_s == 9.0
    )
    # Isolated circuits per endpoint: the dead override opens its own
    # breaker key while the healthy override (and the env default) pass.
    bapp_brk = _TC2(
        api_mod.create_app(
            backend_resolver=_cap_resolver,
            breaker_threshold=2,
            breaker_cooldown_s=60.0,
        )
    )
    dead = {"base_url": "https://dead.example.com/v1", "api_key": "k", "model": "m"}
    dead_body = {"backend": "byok", "messages": [{"role": "user", "content": "x"}], "byok": dead}
    bapp_brk.post("/harness/complete", json=dead_body)
    bapp_brk.post("/harness/complete", json=dead_body)
    open_dead = bapp_brk.post("/harness/complete", json=dead_body)
    ok_live = bapp_brk.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "messages": [{"role": "user", "content": "x"}],
            "byok": ovr,
        },
    )
    out["byok_override_breaker_isolated"] = (
        open_dead.status_code == 503 and ok_live.status_code == 200
    )
    # Idempotency still works with an override on the body — same body
    # replays, a different key inside byok gets the 409.
    idem_byok = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _CleanBackend()))
    k_body = {
        "backend": "byok",
        "messages": [{"role": "user", "content": "x"}],
        "byok": ovr,
    }
    i1 = idem_byok.post("/harness/complete", json=k_body, headers={"Idempotency-Key": "bk1"})
    i2 = idem_byok.post("/harness/complete", json=k_body, headers={"Idempotency-Key": "bk1"})
    i3 = idem_byok.post(
        "/harness/complete",
        json={**k_body, "byok": {**ovr, "api_key": "sk-different"}},
        headers={"Idempotency-Key": "bk1"},
    )
    out["byok_override_idem_contract"] = (
        i1.status_code == 200 and i2.json().get("replayed") is True and i3.status_code == 409
    )
    # limiter counts denials; a public-path request also draws a token
    limited2 = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        rate_limit_rps=20.0,
    )
    lc2 = _TC2(limited2)
    statuses = [lc2.get("/harness/version").status_code for _ in range(23)]
    time.sleep(0.15)
    m = lc2.get("/metrics")
    out["rate_limit_metrics_counts"] = (
        statuses.count(429) >= 2
        and m.status_code == 200
        and m.json()["rate_limited_total"] == statuses.count(429)
    )
    os.environ["FX1_API_RATE_LIMIT_RPS"] = "3"
    try:
        env_app = api_mod.create_app(
            harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
            backend_resolver=lambda *a, **k: _CleanBackend(),
        )
        ec = _TC2(env_app)
        out["rate_limit_env_config"] = [
            ec.get("/harness/version").status_code for _ in range(5)
        ].count(429) >= 1
    finally:
        os.environ.pop("FX1_API_RATE_LIMIT_RPS", None)

    # --- job event stream: SSE status feed until terminal -------------------
    ev_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=1,
        sse_keepalive_s=0.05,
    )
    evc = _TC2(ev_app)
    ev_jid = evc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    ev_resp = evc.get(f"/harness/jobs/{ev_jid}/events")
    ev_frames = [
        _json.loads(ln[len("data: ") :])
        for ln in ev_resp.text.splitlines()
        if ln.startswith("data: ")
    ]
    out["job_events_stream_terminal"] = (
        ev_resp.status_code == 200
        and "text/event-stream" in ev_resp.headers.get("content-type", "")
        and ev_resp.headers.get("cache-control", "") in ("no-cache", "no-store")
        and len(ev_frames) >= 1
        and ev_frames[-1]["status"] == "succeeded"
        and ev_frames[-1]["job_id"] == ev_jid
    )
    out["job_events_unknown_404"] = evc.get("/harness/jobs/nope/events").status_code == 404
    out["job_events_timeout_422"] = (
        evc.get(f"/harness/jobs/{ev_jid}/events?timeout_s=0.5").status_code == 422
    )
    # a blocked worker keeps the next job queued: the stream emits the
    # queued frame, keeps alive on the quiet job, then closes at timeout_s
    ev_app.state.jobs_executor.submit(lambda: time.sleep(3))
    blk_jid = evc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    blk_resp = evc.get(f"/harness/jobs/{blk_jid}/events?timeout_s=1")
    blk_statuses = [
        _json.loads(ln[len("data: ") :])["status"]
        for ln in blk_resp.text.splitlines()
        if ln.startswith("data: ")
    ]
    out["job_events_queued_frame_then_close"] = blk_resp.status_code == 200 and blk_statuses == [
        "queued"
    ]
    out["job_events_keepalive_frames"] = ": keepalive" in blk_resp.text

    # --- batch submit: per-item outcomes, per-item idempotency ---------------
    b_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=4,
    )
    bc = _TC2(b_app)
    b_resp = bc.post(
        "/harness/jobs/batch",
        json={
            "jobs": [
                {"command": "doctor", "idempotency_key": "bk1"},
                {"command": "doctor"},
                {"command": "no-such-command"},
            ]
        },
    )
    b_out = b_resp.json()
    out["jobs_batch_per_item"] = (
        b_resp.status_code == 202
        and b_out["submitted"] == 2
        and b_out["failed"] == 1
        and bool(b_out["jobs"][0].get("job_id"))
        and b_out["jobs"][2]["code"] == "not_found"
    )
    b_replay = bc.post(
        "/harness/jobs/batch",
        json={"jobs": [{"command": "doctor", "idempotency_key": "bk1"}]},
    ).json()
    out["jobs_batch_idem_replay"] = (
        b_replay["submitted"] == 1
        and b_replay["jobs"][0]["replayed"] is True
        and b_replay["jobs"][0]["job_id"] == b_out["jobs"][0]["job_id"]
    )
    b_conflict = bc.post(
        "/harness/jobs/batch",
        json={"jobs": [{"command": "operations", "idempotency_key": "bk1"}]},
    ).json()
    out["jobs_batch_idem_conflict"] = (
        b_conflict["failed"] == 1
        and b_conflict["jobs"][0]["code"] == "conflict"
        and b_conflict["jobs"][0]["job_id"] is None
    )
    out["jobs_batch_empty_422"] = (
        bc.post("/harness/jobs/batch", json={"jobs": []}).status_code == 422
    )
    out["jobs_batch_over_cap_422"] = (
        bc.post("/harness/jobs/batch", json={"jobs": [{"command": "doctor"}] * 65}).status_code
        == 422
    )
    # single-submit idempotency via the body field (header still wins)
    body_key = bc.post(
        "/harness/jobs", json={"command": "doctor", "idempotency_key": "hdr-body"}
    ).json()
    body_key2 = bc.post(
        "/harness/jobs",
        json={"command": "doctor", "idempotency_key": "hdr-body"},
        headers={"Idempotency-Key": "hdr-body"},
    ).json()
    out["jobs_body_idem_replays"] = (
        body_key2["replayed"] is True and body_key2["job_id"] == body_key["job_id"]
    )
    cap_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=1,
    )
    cap_app.state.jobs_executor.submit(lambda: time.sleep(3))
    cc2 = _TC2(cap_app)
    cap_out = cc2.post(
        "/harness/jobs/batch",
        json={"jobs": [{"command": "doctor"}, {"command": "doctor"}]},
    ).json()
    out["jobs_batch_capacity_partial"] = (
        cap_out["submitted"] == 1
        and cap_out["failed"] == 1
        and cap_out["jobs"][1]["code"] == "over_capacity"
    )

    # --- batch receipt verification ---------------------------------------------
    vb = bc.post(
        "/receipts/verify/batch",
        json={"receipts": [{"not": "a receipt"}, {"x": 1}]},
    ).json()
    out["verify_batch_per_item"] = (
        vb["verified"] + vb["failed"] == 2
        and len(vb["results"]) == 2
        and vb["results"][0]["index"] == 0
        and vb["results"][0]["valid"] is False
    )
    out["verify_batch_empty_422"] = (
        bc.post("/receipts/verify/batch", json={"receipts": []}).status_code == 422
    )
    out["verify_batch_over_cap_422"] = (
        bc.post("/receipts/verify/batch", json={"receipts": [{}] * 65}).status_code == 422
    )

    # --- gzip response compression --------------------------------------------
    g_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    gc = _TC2(g_app)
    g_resp = gc.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    out["gzip_compresses_large_response"] = (
        g_resp.status_code == 200
        and g_resp.headers.get("content-encoding") == "gzip"
        and len(g_resp.content) > 4096
    )
    g_small = gc.get("/health", headers={"Accept-Encoding": "gzip"})
    out["gzip_small_response_untouched"] = (
        g_small.status_code == 200 and "content-encoding" not in g_small.headers
    )
    g_plain = gc.get("/openapi.json", headers={"Accept-Encoding": "identity"})
    out["gzip_requires_accept_encoding"] = (
        g_plain.status_code == 200
        and "content-encoding" not in g_plain.headers
        and len(g_plain.content) > 4096
    )
    g_off_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        gzip_min_bytes=0,
    )
    g_off = _TC2(g_off_app).get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    out["gzip_zero_disables"] = "content-encoding" not in g_off.headers
    try:
        api_mod.create_app(
            harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
            backend_resolver=lambda *a, **k: _CleanBackend(),
            gzip_min_bytes=-1,
        )
        out["gzip_negative_rejected"] = False
    except ValueError:
        out["gzip_negative_rejected"] = True

    # --- job lifecycle hygiene --------------------------------------------------
    big = "x" * (1 << 21)
    chatty_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, big, "e" * (1 << 21))),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    cc3 = _TC2(chatty_app)
    jid_big = cc3.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    deadline = time.time() + 10.0
    while time.time() < deadline:
        rec = cc3.get(f"/harness/jobs/{jid_big}").json()
        if rec["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    out["job_result_capped"] = (
        rec["status"] == "succeeded"
        and rec["result"]["stdout_truncated"] is True
        and rec["result"]["stderr_truncated"] is True
        and len(rec["result"]["stdout"].encode()) <= 1 << 20
        and len(rec["result"]["stderr"].encode()) <= 1 << 20
    )
    small_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "tiny", "warn")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
    )
    sc = _TC2(small_app)
    jid_small = sc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    deadline = time.time() + 10.0
    while time.time() < deadline:
        rec2 = sc.get(f"/harness/jobs/{jid_small}").json()
        if rec2["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    out["job_result_small_unflagged"] = (
        rec2["status"] == "succeeded"
        and rec2["result"]["stdout_truncated"] is False
        and rec2["result"]["stdout"] == "tiny"
    )

    # shutdown: TestClient __exit__ runs the lifespan teardown — queued jobs
    # must flip to cancelled (never silently die) and the drain latch set.
    drain_app = api_mod.create_app(
        harness=_Harness(runner=lambda argv, t: (0, "ran", "")),
        backend_resolver=lambda *a, **k: _CleanBackend(),
        max_inflight=1,
    )
    with _TC2(drain_app) as dc:
        drain_app.state.jobs_executor.submit(lambda: time.sleep(3))
        jid_q = dc.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    dead = drain_app.state.job_store.get(jid_q)
    out["shutdown_cancels_queued"] = (
        dead is not None and dead.status == "cancelled" and dead.finished_at is not None
    )
    out["shutdown_sets_drain"] = drain_app.state.metrics.draining.is_set()

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

    # --- content-addressed receipt fetch --------------------------------------
    # A caller cites sha256 in a completion; these routes serve the sealed
    # bytes back by that hash — ETag is the hash itself.
    import tempfile as _tf  # noqa: PLC0415
    from pathlib import Path as _Pt  # noqa: PLC0415

    with _tf.TemporaryDirectory() as td:
        rdir = _Pt(td)
        sha = good["receipt_sha256"]
        (rdir / "probe_receipt.json").write_text(_json.dumps(good))
        rclient = _TC2(api_mod.create_app(receipts_dir=rdir))
        lst = rclient.get("/receipts")
        out["receipts_index_lists"] = (
            lst.status_code == 200
            and lst.json()["count"] == 1
            and lst.json()["items"][0]["sha256"] == sha
            and lst.json()["items"][0]["name"] == "probe_receipt.json"
        )
        got = rclient.get(f"/receipts/{sha}")
        raw = (rdir / "probe_receipt.json").read_bytes()
        out["receipt_fetch_verbatim"] = (
            got.status_code == 200
            and got.content == raw
            and got.json()["receipt_sha256"] == sha
            and got.headers.get("etag") == f'"{sha}"'
            and "immutable" in (got.headers.get("cache-control") or "")
            and got.headers.get("x-fx1-receipt-valid") == "true"
        )
        miss = rclient.get("/receipts/" + "f" * 64)
        out["receipt_fetch_404"] = (
            miss.status_code == 404 and miss.json().get("code") == "receipt_not_found"
        )
        out["receipt_fetch_422_malformed"] = rclient.get("/receipts/zzz").status_code == 422
        nm = rclient.get(f"/receipts/{sha}", headers={"if-none-match": f'"{sha}"'})
        out["receipt_fetch_conditional_304"] = (
            nm.status_code == 304
            and nm.content == b""
            and nm.headers.get("etag") == f'"{sha}"'
            and rclient.get(
                f"/receipts/{sha}", headers={"if-none-match": '"' + "b" * 64 + '"'}
            ).status_code
            == 200
        )
        # staleness key: a receipt written after the first scan is indexed
        (rdir / "second.json").write_text(_json.dumps({**good, "receipt_sha256": "b" * 64}))
        out["receipts_index_refreshes"] = rclient.get("/receipts").json()["count"] == 2

        # --- cited-receipt verification --------------------------------------
        # With a store mounted, a completion may not footnote evidence the
        # server cannot produce: unknown or malformed hashes are 422 across
        # sync, batch, and stream — checked before the breaker/backend call.
        cite = _TC2(
            api_mod.create_app(receipts_dir=rdir, backend_resolver=lambda *a, **k: _CleanBackend())
        )
        out["cite_resolves_200"] = (
            cite.post(
                "/harness/complete",
                json={
                    "backend": "hosted_k3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "receipt_hashes": [sha],
                },
            ).status_code
            == 200
        )
        bad = cite.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": [{"role": "user", "content": "hi"}],
                "receipt_hashes": ["e" * 64],
            },
        )
        out["cite_unknown_422"] = (
            bad.status_code == 422 and bad.json().get("code") == "receipt_not_found"
        )
        out["cite_malformed_422"] = (
            cite.post(
                "/harness/complete",
                json={
                    "backend": "hosted_k3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "receipt_hashes": ["zzz"],
                },
            ).status_code
            == 422
        )
        out["cite_batch_checked"] = (
            cite.post(
                "/harness/complete/batch",
                json={
                    "backend": "hosted_k3",
                    "batch": [[{"role": "user", "content": "hi"}]],
                    "receipt_hashes": ["e" * 64],
                },
            ).status_code
            == 422
        )
        out["cite_stream_checked"] = (
            cite.post(
                "/harness/complete/stream",
                json={
                    "backend": "hosted_k3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "receipt_hashes": ["e" * 64],
                },
            ).status_code
            == 422
        )
    # No store mounted -> citations stay advisory (nothing to check against).
    nostore = _TC2(
        api_mod.create_app(
            receipts_dir="/nonexistent-receipts-dir-zzz",
            backend_resolver=lambda *a, **k: _CleanBackend(),
        )
    )
    out["cite_advisory_no_store"] = (
        nostore.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": [{"role": "user", "content": "hi"}],
                "receipt_hashes": ["e" * 64],
            },
        ).status_code
        == 200
    )
    gone = _TC2(api_mod.create_app(receipts_dir="/nonexistent-receipts-dir-zzz"))
    out["receipt_fetch_store_unavailable"] = (
        gone.get("/receipts/" + "a" * 64).status_code == 503
        and gone.get("/receipts").status_code == 503
        and gone.get("/receipts/" + "a" * 64).json().get("code") == "receipts_unavailable"
    )

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
    from fx1.serve.client import EXPECTED_API_VERSION  # noqa: PLC0415
    from fx1.serve.contract import API_VERSION as _WIRE_VERSION  # noqa: PLC0415

    out["contract_single_source"] = (
        api_mod.API_VERSION == _WIRE_VERSION == EXPECTED_API_VERSION == "1"
    )
    from fastapi.testclient import TestClient as _CapTC  # noqa: PLC0415

    from fx1.harness import HarnessRole as _HarnessRole  # noqa: PLC0415

    # /harness/capabilities — self-describing feature/limit discovery.
    cap = client.get("/harness/capabilities")
    out["capabilities_route_200"] = cap.status_code == 200
    if cap.status_code == 200:
        capj = cap.json()
        out["capabilities_shape"] = all(
            k in capj for k in ("features", "limits", "backends", "roles", "api_version")
        )
        out["capabilities_features"] = all(
            capj["features"].get(f) is True
            for f in ("idempotency", "sse", "webhooks", "batch", "jobs", "drain")
        )
        out["capabilities_roles_cover_registry"] = set(capj["roles"]) == {
            str(r) for r in _HarnessRole
        }
    else:
        out["capabilities_shape"] = False
        out["capabilities_features"] = False
        out["capabilities_roles_cover_registry"] = False
    cap_app = api_mod.create_app(max_inflight=7, job_max=33, idem_max=17, rate_limit_rps=50.0)
    cap2 = _CapTC(cap_app).get("/harness/capabilities").json()
    out["capabilities_limits_reflect_config"] = (
        cap2["limits"]["max_inflight"] == 7.0
        and cap2["limits"]["job_max"] == 33.0
        and cap2["limits"]["idem_max"] == 17.0
        and cap2["limits"]["rate_limit_rps"] == 50.0
    )
    cap3 = _CapTC(api_mod.create_app(rate_limit_rps=0.0)).get("/harness/capabilities").json()
    out["capabilities_limiter_disabled_reports_zero"] = cap3["limits"]["rate_limit_rps"] == 0.0
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

    # Prometheus text exposition: content-negotiated /metrics for a
    # scrape endpoint — Accept: text/plain (or ?format=prom) renders the
    # 0.0.4 text format; unknown formats are a clean 422; the secured
    # surface keeps requiring auth.
    prom = client.get("/metrics", headers={"Accept": "text/plain"})
    body_text = prom.text
    out["prom_accept_negotiated"] = prom.status_code == 200 and prom.headers[
        "content-type"
    ].startswith("text/plain")
    out["prom_requests_series"] = (
        "# TYPE fx1_requests_total counter" in body_text
        and 'fx1_requests_total{status="200"}' in body_text
    )
    out["prom_gauges"] = (
        "# TYPE fx1_inflight gauge" in body_text
        and "fx1_draining 0" in body_text
        and "# TYPE fx1_rate_limited_total counter" in body_text
        and 'fx1_jobs{status="succeeded"}' in body_text
    )
    prom_q = client.get("/metrics?format=prometheus")
    out["prom_query_param"] = prom_q.status_code == 200 and "fx1_uptime_seconds" in prom_q.text
    out["prom_format_422"] = client.get("/metrics?format=xml").status_code == 422
    out["prom_secured_401"] = (
        secured.get("/metrics", headers={"Accept": "text/plain"}).status_code == 401
    )
    prom_json = client.get("/metrics", headers={"Accept": "application/json"})
    out["prom_json_default"] = prom_json.status_code == 200 and prom_json.headers[
        "content-type"
    ].startswith("application/json")

    # --- usage accounting -------------------------------------------------
    # Token counts the endpoint reports ride the response — per-call on
    # sync complete, batch-level delta on batch, absent (not fabricated)
    # when the backend is silent.
    class _UsageBackend(_CleanBackend):
        def __init__(self) -> None:
            super().__init__()
            self.last_usage: dict[str, int] | None = None
            self.total_usage: dict[str, int] = {}

        def complete(self, messages: list[dict[str, str]]) -> str:
            usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
            self.last_usage = usage
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v
            return super().complete(messages)

    ube = _UsageBackend()
    uapp = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: ube))
    u_ok = uapp.post(
        "/harness/complete", json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]}
    )
    out["usage_reported_on_sync"] = u_ok.status_code == 200 and u_ok.json()["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    u_batch = uapp.post(
        "/harness/complete/batch",
        json={
            "backend": "byok",
            "batch": [
                [{"role": "u", "content": "a"}],
                [{"role": "u", "content": "b"}],
            ],
        },
    )
    out["usage_batch_total_is_delta"] = u_batch.status_code == 200 and u_batch.json()[
        "usage_total"
    ] == {"prompt_tokens": 6, "completion_tokens": 10, "total_tokens": 16}
    silent = batch_client.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    out["usage_absent_when_backend_silent"] = (
        silent.status_code == 200 and silent.json()["usage"] is None
    )
    # The pure extractor: non-int values drop, missing/malformed → None.
    import fx1.serve.backends as _be_mod  # noqa: PLC0415

    out["usage_extract_filters"] = (
        _be_mod._extract_usage({"usage": {"prompt_tokens": 3.0, "weird": "no", "neg": -1}})
        == {"prompt_tokens": 3, "neg": -1}
        and _be_mod._extract_usage({}) is None
        and _be_mod._extract_usage({"usage": "broken"}) is None
    )

    # --- completion observability ------------------------------------------
    # Per-backend outcome counters + a latency histogram over attempted
    # model calls — the uapp backend served 1 sync + 2 batch items above.
    um = uapp.get("/metrics").json()
    out["metrics_complete_counters"] = um["complete"]["byok"]["ok"] == 3
    up = uapp.get("/metrics", headers={"Accept": "text/plain"}).text
    out["metrics_latency_histogram"] = (
        'fx1_complete_total{backend="byok",outcome="ok"} 3' in up
        and 'fx1_complete_latency_ms_count{backend="byok"} 3' in up
        and 'fx1_complete_latency_ms_bucket{backend="byok",le="+Inf"} 3' in up
    )
    # gate refusals land in the error outcome, not silently dropped
    dm = dirty.get("/metrics").json()
    out["metrics_complete_error_outcome"] = dm["complete"]["byok"]["error"] >= 1

    _probe_backend_probes(client, uapp, dirty, api_mod, out)
    return out


def _probe_backend_probes(
    client: Any, uapp: Any, dirty: Any, api_mod: Any, out: dict[str, Any]
) -> None:
    import json as _json  # noqa: PLC0415

    from fastapi.testclient import TestClient as _TC2  # noqa: PLC0415

    # Deep health through the real resolver: a live call, not config flags.
    # Bypasses the breaker, never feeds it; verdicts land under
    # ``probe:<name>`` so monitoring never pollutes completion SLOs.
    p_ok = uapp.post("/harness/backends/byok/probe", json={})
    pj = p_ok.json()
    out["probe_ok"] = (
        p_ok.status_code == 200
        and pj["ok"] is True
        and pj["model"] == "fake-0"
        and pj["latency_ms"] >= 0
    )
    p_dirty = dirty.post("/harness/backends/byok/probe", json={})
    out["probe_honesty_refusal"] = (
        p_dirty.status_code == 200
        and p_dirty.json()["ok"] is False
        and p_dirty.json()["error_class"] == "honesty_refusal"
    )

    # unconfigured backend is a verdict, not a wire fault
    def _unconfigured(*a: Any, **k: Any) -> Any:
        raise RuntimeError("BYOK backend is not configured")

    unconf = _TC2(api_mod.create_app(backend_resolver=_unconfigured))
    p_none = unconf.post("/harness/backends/byok/probe", json={})
    out["probe_unconfigured_verdict"] = (
        p_none.status_code == 200
        and p_none.json()["ok"] is False
        and p_none.json()["error_class"] == "backend_unavailable"
    )
    out["probe_unknown_422"] = (
        client.post("/harness/backends/bogus/probe", json={}).status_code == 422
    )
    pm = uapp.get("/metrics").json()
    out["probe_own_series"] = pm["complete"]["probe:byok"]["ok"] >= 1 and (
        "byok" in pm["complete"] and pm["complete"]["byok"]["ok"] == 3
    )

    # gate pre-flight: the honesty gate callable over the wire — a verdict,
    # not a model call. Stays up during drain, never metered.
    g_ok = client.post("/harness/gate/check", json={"text": "the result used bootstrap intervals"})
    g_bad = client.post("/harness/gate/check", json={"text": "we report Sharpe 2.1 out of sample"})
    out["gate_check_clean"] = g_ok.status_code == 200 and g_ok.json()["ok"] is True
    out["gate_check_refusal"] = (
        g_bad.status_code == 200
        and g_bad.json()["ok"] is False
        and "forbidden" in (g_bad.json()["error"] or "")
    )
    out["gate_check_oversize_422"] = (
        client.post("/harness/gate/check", json={"text": "x" * 262145}).status_code == 422
    )

    # usage ledger: backend-reported tokens accumulate per series; the
    # usage_calls counter separates "silent provider" from "zero bill".
    m2 = uapp.get("/metrics").json()["complete"]
    out["usage_metrics_tokens"] = (
        m2["byok"]["total_tokens"] >= 24
        and m2["byok"]["usage_calls"] >= 3
        and m2["byok"]["prompt_tokens"] >= 9
        and m2["probe:byok"]["total_tokens"] >= 8
        and m2["probe:byok"]["usage_calls"] == 1
    )
    prom = uapp.get("/metrics", params={"format": "prom"}).text
    out["usage_prom_lines"] = (
        'fx1_complete_tokens_total{backend="byok",kind="total"}' in prom
        and 'fx1_complete_usage_calls_total{backend="byok"}' in prom
    )

    # streaming usage: a backend that reports usage at stream end lands it
    # on the final SSE frame AND the metrics ledger.
    class _StreamUsage:
        def __init__(self) -> None:
            self.last_usage: dict[str, int] | None = None
            self._model = "stream-usage-0"

        def complete(self, messages: list[dict[str, str]]) -> str:
            return "hello"

        def stream(self, messages: list[dict[str, str]]) -> Any:
            yield "he"
            yield "llo"
            self.last_usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}

    sapp = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: _StreamUsage()))
    s_ok = sapp.post(
        "/harness/complete/stream",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    frames = [
        _json.loads(ln[len("data: ") :])
        for ln in s_ok.text.splitlines()
        if ln.startswith("data: ") and ln[len("data: ") :] != "[DONE]"
    ]
    s_final = next(f for f in frames if f.get("type") == "final")
    s_m = sapp.get("/metrics").json()["complete"]["byok"]
    out["stream_usage_final_frame"] = s_final.get("usage") == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    out["stream_usage_metrics"] = s_m["total_tokens"] == 8 and s_m["usage_calls"] == 1

    # wire parser: a provider chunk carrying ``usage`` (incl. one with no
    # ``choices`` at all) lands in the out-box; malformed stays fatal.
    class _FakeResp:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def __iter__(self) -> Any:
            return iter(self._lines)

    import fx1.serve.backends as _be_mod  # noqa: PLC0415

    wire_frames = [
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n',
        b'data: {"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n',
        b"data: [DONE]\n",
    ]
    import urllib.request as _urlreq  # noqa: PLC0415

    orig_urlopen = _urlreq.urlopen
    _urlreq.urlopen = lambda req, timeout=None: _FakeResp(wire_frames)  # type: ignore[assignment]
    try:
        box: list[dict[str, int]] = []
        toks = list(
            _be_mod._openai_chat_stream(
                "http://wire.test",
                model="m",
                messages=[],
                timeout_s=1.0,
                api_key=None,
                label="t",
                usage_out=box,
            )
        )
    finally:
        _urlreq.urlopen = orig_urlopen
    out["stream_usage_parser"] = toks == ["he"] and box == [
        {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    ]

    # last-probe cache: GET /harness/backends surfaces the most recent
    # verdict so a scrape reads deep health without spending a live call.
    b_u = uapp.get("/harness/backends").json()["byok"]["last_probe"]
    b_d = dirty.get("/harness/backends").json()["byok"]["last_probe"]
    fresh = _TC2(api_mod.create_app(backend_resolver=lambda *a, **k: None))
    out["backends_last_probe"] = (
        b_u is not None
        and b_u["ok"] is True
        and b_u["checked_at"] > 0
        and b_d is not None
        and b_d["ok"] is False
        and b_d["error_class"] == "honesty_refusal"
        and fresh.get("/harness/backends").json()["byok"]["last_probe"] is None
    )

    # completion log: every gated call leaves a fetchable record (hashes,
    # usage, verdict) — the harness's own calls are auditable evidence.
    import hashlib as _hl  # noqa: PLC0415

    c_res = uapp.post(
        "/harness/complete",
        json={"backend": "byok", "messages": [{"role": "u", "content": "x"}]},
    )
    c_id = c_res.json()["completion_id"]
    c_rec = uapp.get(f"/harness/completions/{c_id}")
    want_p = _hl.sha256(
        _json.dumps([{"role": "u", "content": "x"}], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    want_o = _hl.sha256(b"clean:x").hexdigest()
    out["completions_logged_sync"] = (
        c_res.status_code == 200
        and c_res.headers.get("x-fx1-completion-id") == c_id
        and c_rec.status_code == 200
        and c_rec.json()["ok"] is True
        and c_rec.json()["prompt_sha256"] == want_p
        and c_rec.json()["output_sha256"] == want_o
        and c_rec.json()["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    )
    out["completions_get_404"] = (
        uapp.get("/harness/completions/00000000000000000000000000000000").status_code == 404
    )
    d_fail = [
        r
        for r in dirty.get("/harness/completions", params={"backend": "byok"}).json()["items"]
        if r["ok"] is False
    ]
    out["completions_logged_failure"] = (
        len(d_fail) >= 1
        and d_fail[0]["error_class"] == "honesty_refusal"
        and d_fail[0]["output_sha256"] is None
    )
    out["completions_list_filter"] = (
        uapp.get("/harness/completions", params={"backend": "byok"}).json()["count"] >= 4
        and uapp.get("/harness/completions", params={"backend": "local_fx1"}).json()["count"] == 0
        and uapp.get("/harness/completions", params={"limit": 0}).status_code == 422
    )

    s_final_id = s_final.get("completion_id")
    s_rec = sapp.get(f"/harness/completions/{s_final_id}")
    out["completions_logged_stream"] = (
        isinstance(s_final_id, str)
        and s_rec.status_code == 200
        and s_rec.json()["ok"] is True
        and s_rec.json()["output_sha256"] == _hl.sha256(b"hello").hexdigest()
    )

    b_res = uapp.post(
        "/harness/complete/batch",
        json={"backend": "byok", "batch": [[{"role": "u", "content": "q"}]]},
    ).json()
    b_cid = b_res["results"][0]["completion_id"]
    b_rec = uapp.get(f"/harness/completions/{b_cid}")
    out["completions_logged_batch"] = (
        isinstance(b_cid, str)
        and b_rec.status_code == 200
        and b_rec.json()["ok"] is True
        and b_rec.json()["usage"] is None
    )

    # the ring is bounded and newest-first — oldest records evict at cap
    cap3 = api_mod._CompletionLog(cap=3)
    for i in range(5):
        cap3.append(
            api_mod.CompletionRecord(
                completion_id=f"c{i}",
                backend="byok",
                ok=True,
                latency_ms=0.0,
                at=float(i),
                prompt_sha256="p",
            )
        )
    out["completions_ring_cap"] = (
        cap3.get("c0") is None
        and len(cap3.latest(10, None)) == 3
        and [r.completion_id for r in cap3.latest(10, None)] == ["c4", "c3", "c2"]
    )

    # --- sealed per-call receipt export -------------------------------------
    # one logged call exports as a sealed fx1_completion_record.v1 doc:
    # seal re-derives, verify_receipt accepts it, tampering the record's
    # output hash breaks the seal, and exports are byte-deterministic.
    from quant_fund.research.receipt_v2 import (  # noqa: PLC0415
        verify_receipt_payload as _vrp,
    )
    from quant_fund.utils.hashing import (  # noqa: PLC0415
        canonical_json_bytes as _cjb,
    )
    from quant_fund.utils.hashing import (
        hash_bytes as _hb,
    )

    rc = uapp.get(f"/harness/completions/{c_id}/receipt")
    rc_doc = rc.json()
    out["completion_receipt_export"] = (
        rc.status_code == 200
        and rc_doc["kind"] == "fx1_completion_record"
        and rc_doc["schema"] == "fx1_completion_record.v1"
        and rc_doc["record"]["completion_id"] == c_id
        and rc_doc["receipt_sha256"]
        == _hb(_cjb({k: v for k, v in rc_doc.items() if k != "receipt_sha256"}))
    )
    out["completion_receipt_verifies"] = _vrp(rc_doc)["valid"] is True
    out["completion_receipt_verify_route"] = (
        uapp.post("/receipts/verify", json={"receipt": rc_doc}).json().get("valid") is True
    )
    out["completion_receipt_deterministic"] = (
        uapp.get(f"/harness/completions/{c_id}/receipt").json() == rc_doc
    )
    tampered_rec = _json.loads(_json.dumps(rc_doc))
    tampered_rec["record"]["output_sha256"] = "0" * 64
    out["completion_receipt_tamper"] = _vrp(tampered_rec)["valid"] is False
    out["completion_receipt_404"] = (
        uapp.get("/harness/completions/00000000000000000000000000000000/receipt").status_code == 404
    )

    # --- sealed job receipt -------------------------------------------------
    # a terminal job exports as fx1_job_record.v1: streams digested (never
    # content), seal re-derives, verify route accepts, tampering breaks.
    from fx1.harness import Harness as _Harness  # noqa: PLC0415

    jr = _TC2(
        api_mod.create_app(
            harness=_Harness(runner=lambda argv, t: (0, "ran:" + " ".join(argv), ""))
        )
    )
    j_id = jr.post("/harness/jobs", json={"command": "doctor"}).json()["job_id"]
    for _ in range(500):
        if jr.get(f"/harness/jobs/{j_id}").json()["status"] in (
            "succeeded",
            "failed",
            "cancelled",
        ):
            break
        time.sleep(0.01)
    jr_res = jr.get(f"/harness/jobs/{j_id}/receipt")
    jr_doc = jr_res.json()
    want_stdout = _hl.sha256(b"ran:doctor").hexdigest()
    out["job_receipt_export"] = (
        jr_res.status_code == 200
        and jr_doc["kind"] == "fx1_job_record"
        and jr_doc["schema"] == "fx1_job_record.v1"
        and jr_doc["record"]["job_id"] == j_id
        and jr_doc["record"]["status"] == "succeeded"
        and jr_doc["record"]["result"]["stdout_sha256"] == want_stdout
        and "stdout" not in jr_doc["record"]["result"]
        and jr_doc["receipt_sha256"]
        == _hb(_cjb({k: v for k, v in jr_doc.items() if k != "receipt_sha256"}))
    )
    out["job_receipt_verifies"] = (
        _vrp(jr_doc)["valid"] is True
        and jr.post("/receipts/verify", json={"receipt": jr_doc}).json().get("valid") is True
    )
    out["job_receipt_deterministic"] = jr.get(f"/harness/jobs/{j_id}/receipt").json() == jr_doc
    tampered_job = _json.loads(_json.dumps(jr_doc))
    tampered_job["record"]["status"] = "failed"
    out["job_receipt_tamper"] = _vrp(tampered_job)["valid"] is False
    out["job_receipt_404"] = jr.get("/harness/jobs/nope/receipt").status_code == 404


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
            "payload, auth is X-API-Key or loopback-only, and every gated "
            "call lands in the bounded completion log (X-Fx1-Completion-Id "
            "handle; hashes, usage, verdict — never content) fetchable via "
            "GET /harness/completions[/{id}], and each logged call exports "
            "as a sealed fx1_completion_record.v1 document "
            "(GET …/{id}/receipt) — deterministic, verifiable through "
            "verify_receipt / POST /receipts/verify, and broken by any "
            "byte of record tampering. Terminal jobs export the same way "
            "as fx1_job_record.v1 (GET /harness/jobs/{id}/receipt): "
            "stdout/stderr digested inside record.result, callback URL "
            "hashed, seal re-derives and verifies, tampering breaks it."
            if ok
            else f"HARNESS API AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
