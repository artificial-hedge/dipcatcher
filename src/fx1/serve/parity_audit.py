"""Surface-parity audit — the SDK, the HTTP API, and HarnessClient are one contract.

``fx1.sdk.Fx1Harness`` (in-process), ``fx1.serve.api`` (the wire), and
``fx1.serve.client.HarnessClient`` (the remote caller) expose the same
harness over three transports. This audit drives the SAME injected backend
through all three and asserts byte-identical results — a drift between
what an in-process caller, the server, and a remote client get is a
product defect, not a transport detail.

- *Complete parity* — ``Fx1Harness.complete`` output equals
  ``POST /harness/complete``: identical content (evidence footer included),
  identical ``model``/``backend``/``receipt_hashes`` envelope fields.
- *Batch parity* — ``complete_many`` results equal
  ``POST /harness/complete/batch`` items in submission order, one shared
  backend per batch on both surfaces (close counts prove it).
- *Stream parity* — SSE ``token`` frames concatenate to exactly the
  ``complete`` content; ``stream_complete`` returns the identical chunk
  list; the ``final`` envelope matches the completion envelope.
- *Error parity* — the same fault surfaces with the same class on both:
  gate refusal (SDK raises ``Fx1HonestyError`` / API 502 JSON with the
  refusal text), non-streaming backend (NotImplementedError / 501),
  checkpoint_dir on a non-local backend (ValueError / 422), unknown
  backend (KeyError / 422 literal rejection), malformed receipts.
- *Verifier parity* — ``verify_receipt`` and ``POST /receipts/verify``
  agree on valid/errors/warnings/schema for a sealed receipt, a tampered
  digest, and a non-receipt dict.
- *Registry/health parity* — command list and backend presence booleans
  are identical over both surfaces.
- *Remote-client parity* — ``HarnessClient`` (urllib transport injectable)
  returns the SDK's own result types; wire status codes map back to the
  SDK's exception classes (KeyError/ValueError/BackendNotConfiguredError/
  NotImplementedError/Fx1HonestyError), auth faults are
  ``HarnessAuthError``, transport faults ``HarnessTransportError``.
- *In-flight cap* — ``max_inflight`` (env ``FX1_API_MAX_INFLIGHT``, default
  16) bounds concurrent heavy requests: saturation is an honest 503 with
  ``Retry-After``, cheap routes stay responsive, the slot releases
  cleanly, ``max_inflight<1`` fails app construction.

Sealed ``parity_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["parity_audit", "parity_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_BYOK_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")
_LOCAL_ENVS = (
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
)


class _ParityBackend:
    """Deterministic streaming backend — one instance per resolution."""

    _closed_total = 0

    def __init__(self) -> None:
        self._model = "parity-v0"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return f"echo:{messages[-1]['content']}"

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        text = self.complete(messages)
        yield text[:3]
        yield text[3:]

    def close(self) -> None:
        _ParityBackend._closed_total += 1


class _DirtyBackend(_ParityBackend):
    def complete(self, messages: list[dict[str, str]]) -> str:
        return "total Sharpe 4.2 on NAV"  # forbidden headline

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield "total Sharpe 4.2 on NAV"


class _NonStreamingBackend:
    def __init__(self) -> None:
        self._model = "ns-v0"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "ans"

    def close(self) -> None:
        return None


class _FlakyBackend(_ParityBackend):
    """Refuses (gate-tripping output) only on prompts containing 'bad'."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        if "bad" in messages[-1]["content"]:
            return "total Sharpe 4.2 on NAV"
        return super().complete(messages)


def _raises(fn: Any) -> tuple[str, str]:
    """(exception class name, str(exc)) — ("", "") when no raise."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — probe captures the class
        return type(exc).__name__, str(exc)
    return "", ""


def _surfaces(
    backend: Any,
) -> tuple[Any, TestClient]:
    """(Fx1Harness, TestClient) wired to one resolver returning `backend`'s class."""
    from fastapi.testclient import TestClient as _TC

    import fx1.serve.api as api_mod
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, f"ran:{' '.join(argv)}", ""

    def resolver(name: str, **kw: Any) -> Any:
        if name not in {"hosted_k3", "local_fx1", "byok"}:
            raise KeyError(name)
        return backend()

    sdk = Fx1Harness(harness=Harness(runner=fake_runner), backend_resolver=resolver)
    app = api_mod.create_app(harness=Harness(runner=fake_runner), backend_resolver=resolver)
    return sdk, _TC(app)


def _tc_transport(client: TestClient) -> Any:
    """Adapt HarnessClient's transport contract to a TestClient."""

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        p = urllib.parse.urlparse(url)
        path = p.path + (f"?{p.query}" if p.query else "")
        resp = (
            client.get(path, headers=headers)
            if method == "GET"
            else client.post(path, json=payload, headers=headers)
        )
        return resp.status_code, dict(resp.headers), resp.content

    return send


def parity_audit() -> dict[str, bool]:
    out: dict[str, bool] = {}
    saved = {
        k: os.environ.get(k) for k in (_API_KEY_ENV, *_BYOK_ENVS, *_LOCAL_ENVS, "MOONSHOT_API_KEY")
    }
    try:
        os.environ.pop(_API_KEY_ENV, None)

        sdk, client = _surfaces(_ParityBackend)
        msg = [{"role": "user", "content": "ping"}]
        receipt = "a" * 64
        sdk_out = sdk.complete(msg, backend="byok", receipt_hashes=[receipt])
        api_out = client.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": msg,
                "receipt_hashes": [receipt],
            },
        )
        out["complete_status_200"] = api_out.status_code == 200
        api_json = api_out.json()
        out["complete_content_identical"] = api_json["content"] == sdk_out.content
        out["complete_envelope_identical"] = (
            api_json["backend"] == sdk_out.backend
            and api_json["model"] == sdk_out.model
            and tuple(api_json["receipt_hashes"]) == sdk_out.receipt_hashes
        )
        out["complete_footer_identical"] = (
            sdk_out.content.endswith(
                f"Evidence: `{receipt[:16]}…` — verify with `dipcatcher verify-research`."
            )
            and api_json["content"] == sdk_out.content
        )

        # --- batch parity -----------------------------------------------------
        _ParityBackend._closed_total = 0
        batch = [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
        sdk_many = sdk.complete_many(batch, backend="byok", receipt_hashes=[receipt], max_workers=2)
        api_many = client.post(
            "/harness/complete/batch",
            json={
                "backend": "byok",
                "batch": batch,
                "receipt_hashes": [receipt],
                "max_workers": 2,
            },
        )
        out["batch_status_200"] = api_many.status_code == 200
        api_items = api_many.json()
        out["batch_content_identical"] = [r.content for r in sdk_many] == [
            i["content"] for i in api_items["results"]
        ] and all(i["ok"] for i in api_items["results"])
        out["batch_envelope_identical"] = (
            api_items["backend"] == sdk_many[0].backend
            and api_items["model"] == sdk_many[0].model
            and tuple(api_items["receipt_hashes"]) == sdk_many[0].receipt_hashes
        )
        out["batch_one_backend_each_side"] = _ParityBackend._closed_total == 2

        # --- stream parity ----------------------------------------------------
        sdk_chunks = sdk.stream_complete(msg, backend="byok", receipt_hashes=[receipt])
        api_stream = client.post(
            "/harness/complete/stream",
            json={
                "backend": "byok",
                "messages": msg,
                "receipt_hashes": [receipt],
            },
        )
        out["stream_status_200_sse"] = api_stream.status_code == 200 and api_stream.headers[
            "content-type"
        ].startswith("text/event-stream")
        frames = [
            ln[len("data: ") :] for ln in api_stream.text.splitlines() if ln.startswith("data: ")
        ]
        payloads = [json.loads(f) for f in frames if f.strip() != "[DONE]"]
        token_chunks = [p["content"] for p in payloads if p["type"] == "token"]
        final = [p for p in payloads if p["type"] == "final"]
        out["stream_chunks_identical"] = token_chunks == sdk_chunks
        out["stream_joined_eq_complete"] = (
            "".join(token_chunks) == sdk_out.content and "".join(sdk_chunks) == sdk_out.content
        )
        out["stream_done_terminal"] = frames[-1].strip() == "[DONE]"
        out["stream_final_envelope"] = (
            len(final) == 1
            and final[0]["model"] == sdk_out.model
            and final[0]["receipt_hashes"] == [receipt]
        )

        # --- error parity -----------------------------------------------------
        dirty_sdk, dirty_api = _surfaces(_DirtyBackend)
        sdk_err_cls, sdk_err_txt = _raises(lambda: dirty_sdk.complete(msg, backend="byok"))
        api_err = dirty_api.post("/harness/complete", json={"backend": "byok", "messages": msg})
        out["gate_sdk_raises_api_502"] = (
            sdk_err_cls == "Fx1HonestyError"
            and api_err.status_code == 502
            and "honesty gate" in api_err.json()["detail"]
        )
        # the API surfaces the SDK's exception text verbatim after the prefix
        out["gate_error_text_shared"] = (
            api_err.status_code == 502
            and api_err.json()["detail"] == f"honesty gate refused model output: {sdk_err_txt}"
        )
        sdk_serr, _ = _raises(lambda: dirty_sdk.stream_complete(msg, backend="byok"))
        api_serr = dirty_api.post(
            "/harness/complete/stream", json={"backend": "byok", "messages": msg}
        )
        out["stream_gate_parity"] = (
            sdk_serr == "Fx1HonestyError"
            and api_serr.status_code == 502
            and "total Sharpe" not in api_serr.text
        )

        ns_sdk, ns_api = _surfaces(_NonStreamingBackend)
        out["stream_unsupported_parity"] = (
            _raises(lambda: ns_sdk.stream_complete(msg, backend="byok"))[0] == "NotImplementedError"
            and ns_api.post(
                "/harness/complete/stream",
                json={"backend": "byok", "messages": msg},
            ).status_code
            == 501
        )
        out["checkpoint_dir_non_local_parity"] = (
            _raises(lambda: sdk.complete(msg, backend="byok", checkpoint_dir="/x"))[0]
            == "ValueError"
            and client.post(
                "/harness/complete",
                json={
                    "backend": "byok",
                    "messages": msg,
                    "checkpoint_dir": "/x",
                },
            ).status_code
            == 422
        )
        # Resolver KeyError is unreachable over the wire: CompleteRequest's
        # backend Literal rejects unknown names with 422 before resolution;
        # the SDK raises KeyError on the same name. Both fail closed.
        out["unknown_backend_fail_closed"] = (
            _raises(lambda: sdk.complete(msg, backend="bogus"))[0] == "KeyError"
            and client.post(
                "/harness/complete",
                json={"backend": "bogus", "messages": msg},
            ).status_code
            == 422
        )
        out["empty_batch_contract"] = (
            sdk.complete_many([], backend="byok") == []
            and client.post(
                "/harness/complete/batch",
                json={"backend": "byok", "batch": []},
            ).status_code
            == 422
        )

        # --- verifier parity ----------------------------------------------------
        from fx1.serve.byok_audit import byok_audit_bench

        good = byok_audit_bench()
        sdk_v = sdk.verify_receipt(good)
        api_v = client.post("/receipts/verify", json={"receipt": good}).json()
        out["verify_valid_parity"] = (
            sdk_v.valid is True
            and api_v["valid"] is True
            and api_v["schema_tag"] == sdk_v.schema_tag
            and api_v["kind"] == sdk_v.kind
            and api_v["verdict"] == sdk_v.verdict
            and tuple(api_v["errors"]) == sdk_v.errors
            and tuple(api_v["warnings"]) == sdk_v.warnings
        )
        tampered = dict(good)
        tampered["receipt_sha256"] = "0" * 64
        sdk_tv = sdk.verify_receipt(tampered)
        api_tv = client.post("/receipts/verify", json={"receipt": tampered}).json()
        out["verify_tamper_parity"] = (
            sdk_tv.valid is False
            and api_tv["valid"] is False
            and tuple(api_tv["errors"]) == sdk_tv.errors
        )
        sdk_mv = sdk.verify_receipt({"not": "a receipt"})
        api_mv = client.post("/receipts/verify", json={"receipt": {"not": "a receipt"}}).json()
        out["verify_malformed_parity"] = (
            sdk_mv.valid is False
            and api_mv["valid"] is False
            and tuple(api_mv["errors"]) == sdk_mv.errors
        )

        # --- registry / run / health parity -----------------------------------
        api_cmds = client.get("/harness/commands").json()
        out["commands_parity"] = sorted(sdk.commands()) == sorted(
            c["name"] for c in api_cmds["items"]
        ) and api_cmds["total"] == len(api_cmds["items"])
        name = sorted(sdk.commands())[0]
        sdk_run = sdk.run(name)
        api_run = client.post("/harness/runs", json={"command": name}).json()
        out["run_parity"] = (
            api_run["exit_code"] == sdk_run.exit_code
            and api_run["stdout"] == sdk_run.stdout
            and api_run["ok"] == sdk_run.ok
        )
        api_health = client.get("/health").json()
        sdk_health = sdk.health()
        out["health_parity"] = (
            api_health["backends"] == sdk_health.backends
            and api_health["version"] == sdk_health.version
            and api_health["registered_commands"] == sdk_health.registered_commands
        )

        # --- HarnessClient: the remote-caller surface --------------------------------
        from fx1.serve.client import (
            HarnessAuthError,
            HarnessClient,
            HarnessJobError,
            HarnessTransportError,
        )

        remote = HarnessClient("http://harness.test", transport=_tc_transport(client))
        rem_out = remote.complete(msg, backend="byok", receipt_hashes=[receipt])
        out["client_complete_identical"] = rem_out == sdk_out
        out["client_stream_identical"] = (
            remote.stream_complete(msg, backend="byok", receipt_hashes=[receipt]) == sdk_chunks
        )
        out["client_batch_identical"] = [
            r.content
            for r in remote.complete_many(
                batch, backend="byok", receipt_hashes=[receipt], max_workers=2
            )
        ] == [r.content for r in sdk_many]
        rem_v = remote.verify_receipt(good)
        out["client_verify_parity"] = (
            rem_v.valid is True
            and rem_v.schema_tag == sdk_v.schema_tag
            and rem_v.errors == sdk_v.errors
        )
        rem_tv = remote.verify_receipt(tampered)
        out["client_verify_tamper_parity"] = (
            rem_tv.valid is False and rem_tv.errors == sdk_tv.errors
        )
        out["client_commands_parity"] = sorted(remote.commands()) == sorted(sdk.commands())
        rem_run = remote.run(name)
        out["client_run_parity"] = (
            rem_run.exit_code == sdk_run.exit_code
            and rem_run.stdout == sdk_run.stdout
            and rem_run.ok == sdk_run.ok
        )
        rem_health = remote.health()
        out["client_health_parity"] = (
            rem_health.backends == sdk_health.backends and rem_health.version == sdk_health.version
        )
        filtered = remote.commands(role="evaluation")
        out["client_role_filter"] = 0 < len(filtered) <= len(remote.commands()) and set(
            filtered
        ) <= set(remote.commands())
        # error mapping: the wire's codes map back to the SDK's classes
        dirty_remote = HarnessClient("http://harness.test", transport=_tc_transport(dirty_api))
        out["client_gate_maps_fx1honesty"] = (
            _raises(lambda: dirty_remote.complete(msg, backend="byok"))[0] == "Fx1HonestyError"
        )
        out["client_stream_gate_maps"] = (
            _raises(lambda: dirty_remote.stream_complete(msg, backend="byok"))[0]
            == "Fx1HonestyError"
        )
        ns_remote = HarnessClient("http://harness.test", transport=_tc_transport(ns_api))
        out["client_501_maps"] = (
            _raises(lambda: ns_remote.stream_complete(msg, backend="byok"))[0]
            == "NotImplementedError"
        )
        out["client_404_maps_keyerror"] = (
            _raises(lambda: remote.run("no-such-command"))[0] == "KeyError"
        )
        out["client_422_maps_valueerror"] = (
            _raises(lambda: remote.complete(msg, backend="bogus"))[0] == "ValueError"
        )
        # per-item batch failure: lowest-index gate refusal raises on the client
        flaky_sdk, flaky_api = _surfaces(_FlakyBackend)
        flaky_remote = HarnessClient("http://harness.test", transport=_tc_transport(flaky_api))
        mixed = [
            [{"role": "user", "content": "fine"}],
            [{"role": "user", "content": "bad"}],
        ]
        out["client_batch_gate_maps"] = (
            _raises(lambda: flaky_remote.complete_many(mixed, backend="byok"))[0]
            == "Fx1HonestyError"
        )
        out["sdk_batch_gate_raises"] = (
            _raises(lambda: flaky_sdk.complete_many(mixed, backend="byok"))[0] == "Fx1HonestyError"
        )

        # transport-level faults
        def _dead_transport(*a: Any) -> Any:
            raise HarnessTransportError("connection refused")

        out["client_transport_unreachable"] = (
            _raises(
                lambda: HarnessClient("http://harness.test", transport=_dead_transport).health()
            )[0]
            == "HarnessTransportError"
        )
        out["client_auth_error_maps"] = (
            _raises(
                lambda: HarnessClient(
                    "http://harness.test",
                    transport=lambda *a: (401, {}, b'{"detail":"denied"}'),
                ).health()
            )[0]
            == HarnessAuthError.__name__
        )
        out["client_base_url_fail_closed"] = all(
            _raises(lambda u=u: HarnessClient(u))[0] == "ValueError"
            for u in ("not-a-url", "ftp://x", "http://")
        )
        out["client_timeout_fail_closed"] = (
            _raises(lambda: HarnessClient("http://h.test", timeout_s=0))[0] == "ValueError"
        )
        captured: list[dict[str, str]] = []

        def _capture(method: str, url: str, payload: Any, headers: dict[str, str], t: float) -> Any:
            captured.append(headers)
            return (200, {}, b'{"status":"ok","version":"v","registered_commands":0,"backends":{}}')

        HarnessClient("http://harness.test", api_key="probe-key", transport=_capture).health()
        out["client_api_key_header"] = (
            bool(captured) and captured[0].get("X-API-Key") == "probe-key"
        )
        # stream without terminal [DONE] is a transport fault, never data
        out["client_stream_no_done_fails"] = (
            _raises(
                lambda: HarnessClient(
                    "http://harness.test",
                    transport=lambda *a: (
                        200,
                        {},
                        b'data: {"type":"token","content":"x"}\n\n',
                    ),
                ).stream_complete(msg, backend="byok")
            )[0]
            == "HarnessTransportError"
        )

        # --- in-flight cap: saturation is an honest 503, never a queue hang -----
        import fx1.serve.api as api_mod
        from fx1.harness import Harness

        def fake_runner2(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
            return 0, "ran", ""

        def resolver2(name: str, **kw: Any) -> Any:
            return _ParityBackend()

        capped_app = api_mod.create_app(
            harness=Harness(runner=fake_runner2),
            backend_resolver=resolver2,
            max_inflight=2,
        )
        from fastapi.testclient import TestClient as _TC

        capped = _TC(capped_app)
        slots = capped_app.state.inflight_slots
        out["cap_normal_request_passes"] = (
            capped.post(
                "/harness/complete",
                json={"backend": "byok", "messages": msg},
            ).status_code
            == 200
        )
        held1 = slots.acquire(blocking=False)
        held2 = slots.acquire(blocking=False)
        saturated = capped.post(
            "/harness/complete",
            json={"backend": "byok", "messages": msg},
        )
        out["cap_saturated_503"] = (
            held1
            and held2
            and saturated.status_code == 503
            and "max_inflight" in saturated.json()["detail"]
        )
        out["cap_client_maps_503"] = (
            _raises(
                lambda: HarnessClient(
                    "http://harness.test", transport=_tc_transport(capped)
                ).complete(msg, backend="byok")
            )[0]
            == "BackendNotConfiguredError"
        )
        out["cap_runs_also_capped"] = (
            capped.post("/harness/runs", json={"command": name}).status_code == 503
        )
        out["cap_cheap_routes_respond"] = (
            capped.get("/health").status_code == 200
            and capped.get("/harness/commands").status_code == 200
        )
        if held1:
            slots.release()
        if held2:
            slots.release()
        out["cap_released_recovers"] = (
            capped.post(
                "/harness/complete",
                json={"backend": "byok", "messages": msg},
            ).status_code
            == 200
        )
        out["cap_invalid_config_fails"] = (
            _raises(
                lambda: api_mod.create_app(harness=Harness(runner=fake_runner2), max_inflight=0)
            )[0]
            == "ValueError"
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # --- resilience: bounded retries + Retry-After honoring -----------------
    from fx1.serve.client import HarnessTransportError  # noqa: PLC0415

    def _scripted(
        seq: list[Any],
    ) -> tuple[Any, dict[str, int]]:
        calls = {"n": 0}

        def transport(
            method: str,
            url: str,
            payload: Any,
            headers: Any,
            timeout_s: float,
        ) -> tuple[int, Mapping[str, str], bytes]:
            item = seq[min(calls["n"], len(seq) - 1)]
            calls["n"] += 1
            if item == "RAISE":
                raise HarnessTransportError("boom")
            status, hdrs, body = item
            return status, hdrs, body

        return transport, calls

    import json as _json_mod  # noqa: PLC0415

    def _health_body() -> bytes:
        return _json_mod.dumps(
            {"status": "ok", "version": "v", "registered_commands": 1, "backends": {}}
        ).encode()

    sleeps: list[float] = []
    tr, calls = _scripted(["RAISE", "RAISE", (200, {}, _health_body())])
    resilient = HarnessClient(
        "http://h.test",
        transport=tr,
        max_retries=2,
        sleep=sleeps.append,
    )
    out["retry_get_recovers"] = (
        resilient.health().status == "ok" and calls["n"] == 3 and len(sleeps) == 2
    )

    sleeps.clear()
    tr2, calls2 = _scripted(
        [
            (503, {"Retry-After": "0.25"}, _json_mod.dumps({"detail": "cap"}).encode()),
            (200, {}, _health_body()),
        ]
    )
    resilient2 = HarnessClient("http://h.test", transport=tr2, max_retries=2, sleep=sleeps.append)
    out["retry_after_honored"] = (
        resilient2.health().status == "ok" and calls2["n"] == 2 and sleeps[0] == 0.25
    )

    tr3, calls3 = _scripted(
        [(503, {"Retry-After": "999"}, _json_mod.dumps({"detail": "cap"}).encode())]
    )
    resilient3 = HarnessClient("http://h.test", transport=tr3, max_retries=3, sleep=lambda s: None)
    out["retry_over_budget_fails_fast"] = (
        _raises(lambda: resilient3.health())[0] == "HarnessTransportError" and calls3["n"] == 1
    )

    tr4, calls4 = _scripted([(401, {}, _json_mod.dumps({"detail": "bad key"}).encode())])
    resilient4 = HarnessClient("http://h.test", transport=tr4, max_retries=3, sleep=lambda s: None)
    out["no_retry_on_auth"] = (
        _raises(lambda: resilient4.health())[0] == "HarnessAuthError" and calls4["n"] == 1
    )

    tr5, calls5 = _scripted(["RAISE"])
    resilient5 = HarnessClient("http://h.test", transport=tr5, max_retries=3, sleep=lambda s: None)
    out["writes_never_retry_by_default"] = (
        _raises(lambda: resilient5.complete([{"role": "user", "content": "hi"}]))[0]
        == "HarnessTransportError"
        and calls5["n"] == 1
    )

    verdict_body = _json_mod.dumps(
        {
            "valid": True,
            "path": "<remote>",
            "schema_tag": "x",
            "kind": "k",
            "verdict": "ok",
            "digest_convention": None,
            "errors": [],
            "warnings": [],
        }
    ).encode()
    tr6, calls6 = _scripted(["RAISE", (200, {}, verdict_body)])
    resilient6 = HarnessClient("http://h.test", transport=tr6, max_retries=1, sleep=lambda s: None)
    out["verify_is_idempotent_retried"] = (
        resilient6.verify_receipt({"x": 1}).valid is True and calls6["n"] == 2
    )

    complete_body = _json_mod.dumps(
        {"backend": "byok", "model": "m", "content": "c", "receipt_hashes": []}
    ).encode()
    tr6b, calls6b = _scripted(["RAISE", (200, {}, complete_body)])
    resilient6b = HarnessClient(
        "http://h.test",
        transport=tr6b,
        max_retries=1,
        retry_writes=True,
        sleep=lambda s: None,
    )
    out["writes_retry_when_opted_in"] = (
        resilient6b.complete([{"role": "user", "content": "hi"}]).content == "c"
        and calls6b["n"] == 2
    )

    tr7, calls7 = _scripted(
        [(503, {}, _json_mod.dumps({"detail": "backend not configured"}).encode())]
    )
    resilient7 = HarnessClient("http://h.test", transport=tr7, max_retries=3, sleep=lambda s: None)
    out["unconfigured_503_not_retried"] = (
        _raises(lambda: resilient7.health())[0] == "BackendNotConfiguredError" and calls7["n"] == 1
    )

    out["retry_validation_failclosed"] = all(
        _raises(lambda kw=kw: HarnessClient("http://h.test", **kw))[0] == "ValueError"
        for kw in (
            {"max_retries": -1},
            {"retry_backoff_s": 0.0},
            {"max_retry_wait_s": 0.0},
            {"circuit_breaker_threshold": -1},
            {"circuit_reset_s": 0.0},
        )
    )

    # circuit breaker: transport faults open it; half-open probe closes it
    tr8, calls8 = _scripted(["RAISE"])
    t8 = [0.0]
    cb = HarnessClient(
        "http://h.test",
        transport=tr8,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
        clock=lambda: t8[0],
        sleep=lambda s: None,
    )
    for _ in range(2):
        _raises(lambda: cb.health())
    out["circuit_opens_and_fails_fast"] = (
        _raises(lambda: cb.health())[0] == "HarnessTransportError"
        and calls8["n"] == 2  # the third call never reached the wire
        and "circuit open" in str(_raises(lambda: cb.health())[1])
    )
    # half-open probe: advance the clock past the reset window; a healthy
    # transport closes the circuit
    tr9, calls9 = _scripted(["RAISE", "RAISE", (200, {}, _health_body())])
    t9 = [0.0]
    cb9 = HarnessClient(
        "http://h.test",
        transport=tr9,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
        clock=lambda: t9[0],
        sleep=lambda s: None,
    )
    for _ in range(2):
        _raises(lambda: cb9.health())
    t9[0] = 31.0
    out["circuit_half_open_closes"] = cb9.health().status == "ok" and calls9["n"] == 3
    # a failed half-open probe re-opens the window
    tr10, calls10 = _scripted(["RAISE"])
    t10 = [0.0]
    cb10 = HarnessClient(
        "http://h.test",
        transport=tr10,
        circuit_breaker_threshold=1,
        circuit_reset_s=30.0,
        clock=lambda: t10[0],
        sleep=lambda s: None,
    )
    _raises(lambda: cb10.health())  # trips (threshold 1)
    t10[0] = 31.0
    _raises(lambda: cb10.health())  # half-open probe fails -> re-open
    out["circuit_half_open_reopens"] = calls10["n"] == 2 and "circuit open" in str(
        _raises(lambda: cb10.health())[1]
    )
    # disabled by default: every call reaches the wire
    tr11, calls11 = _scripted(["RAISE"])
    cb11 = HarnessClient("http://h.test", transport=tr11, sleep=lambda s: None)
    for _ in range(4):
        _raises(lambda: cb11.health())
    out["circuit_disabled_by_default"] = calls11["n"] == 4

    # keepalive comments are skipped; in-band error frames map through the
    # same exception table as HTTP errors (keepalive-mode wire contract)
    ka_body = (
        b": keepalive\n\n"
        b'data: {"type":"token","content":"ka-tok"}\n\n'
        b'data: {"type":"final","model":null,"receipt_hashes":[]}\n\n'
        b"data: [DONE]\n\n"
    )
    trka, _ = _scripted([(200, {}, ka_body)])
    out["client_stream_skips_comments"] = HarnessClient(
        "http://harness.test", transport=trka
    ).stream_complete(msg, backend="byok") == ["ka-tok"]
    inband_body = (
        b": keepalive\n\n"
        b'data: {"type":"error","status":503,"detail":"no backend"}\n\n'
        b"data: [DONE]\n\n"
    )
    trin, _ = _scripted([(200, {}, inband_body)])
    out["client_stream_inband_error_maps"] = (
        _raises(
            lambda: HarnessClient("http://harness.test", transport=trin).stream_complete(
                msg, backend="byok"
            )
        )[0]
        == "BackendNotConfiguredError"
    )
    honesty_body = (
        b'data: {"type":"error","status":502,'
        b'"detail":"honesty gate refused model output: sharpe"}\n\n'
        b"data: [DONE]\n\n"
    )
    trhon, _ = _scripted([(200, {}, honesty_body)])
    out["client_stream_inband_honesty_maps"] = (
        _raises(
            lambda: HarnessClient("http://harness.test", transport=trhon).stream_complete(
                msg, backend="byok"
            )
        )[0]
        == "Fx1HonestyError"
    )

    # idempotency: run() sends Idempotency-Key, stable across retries;
    # caller-supplied keys pass through verbatim.
    sent_headers: list[dict[str, str]] = []
    run_body = _json_mod.dumps(
        {
            "command": "doctor",
            "exit_code": 0,
            "stdout": "s",
            "stderr": "",
            "ok": True,
            "timeout_s": 1,
            "replayed": True,
        }
    ).encode()

    def _cap_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        sent_headers.append(dict(headers))
        if len(sent_headers) == 1:
            raise HarnessTransportError("boom")
        return 200, {}, run_body

    c_idem = HarnessClient(
        "http://harness.test",
        transport=_cap_transport,
        max_retries=1,
        sleep=lambda _s: None,
    )
    c_idem.run("doctor")
    out["client_run_idem_stable_across_retry"] = (
        len(sent_headers) == 2
        and bool(sent_headers[0].get("Idempotency-Key"))
        and sent_headers[0]["Idempotency-Key"] == sent_headers[1]["Idempotency-Key"]
    )
    sent_headers.clear()
    c_idem.run("doctor", idempotency_key="explicit-k")
    out["client_run_idem_explicit_key"] = sent_headers[0].get("Idempotency-Key") == "explicit-k"

    # ---- async jobs ----------------------------------------------------------
    submit_body = _json_mod.dumps(
        {"job_id": "abc123", "status": "queued", "replayed": False}
    ).encode()
    status_body = _json_mod.dumps(
        {
            "job_id": "abc123",
            "status": "succeeded",
            "created_at": 1.0,
            "finished_at": 2.0,
            "result": {
                "command": "doctor",
                "exit_code": 0,
                "stdout": "s",
                "stderr": "",
                "ok": True,
                "timeout_s": 1,
                "replayed": False,
            },
            "error": None,
        }
    ).encode()
    failed_body = _json_mod.dumps(
        {
            "job_id": "abc123",
            "status": "failed",
            "created_at": 1.0,
            "finished_at": 2.0,
            "result": None,
            "error": "RuntimeError: boom",
        }
    ).encode()
    job_calls: list[tuple[str, str, dict[str, str]]] = []

    def _job_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        job_calls.append((method, url, dict(headers)))
        if url.endswith("/harness/jobs"):
            return 202, {}, submit_body
        return 200, {}, status_body

    c_jobs = HarnessClient(
        "http://harness.test",
        transport=_job_transport,
        sleep=lambda _s: None,
    )
    jid = c_jobs.submit_run("doctor", idempotency_key="jk")
    result = c_jobs.wait_run(jid, poll_s=0.01)
    out["client_submit_returns_job_id"] = jid == "abc123"
    out["client_wait_run_polls_to_result"] = result.command == "doctor" and result.ok
    out["client_job_idem_sent"] = job_calls[0][2].get("Idempotency-Key") == "jk"
    out["client_job_status_get"] = any(
        m == "GET" and "/harness/jobs/abc123" in u for m, u, _h in job_calls
    )
    job_calls.clear()

    def _fail_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return 200, {}, failed_body

    c_fail = HarnessClient(
        "http://harness.test",
        transport=_fail_transport,
        sleep=lambda _s: None,
    )
    try:
        c_fail.wait_run("abc123", poll_s=0.01)
        out["client_wait_run_failed_raises"] = False
    except HarnessJobError:
        out["client_wait_run_failed_raises"] = True

    def _queued_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return (
            200,
            {},
            _json_mod.dumps(
                {
                    "job_id": "abc123",
                    "status": "queued",
                    "created_at": 1.0,
                    "finished_at": None,
                    "result": None,
                    "error": None,
                }
            ).encode(),
        )

    _ticks = iter([t * 0.03 for t in range(200)])
    c_wait = HarnessClient(
        "http://harness.test",
        transport=_queued_transport,
        sleep=lambda _s: None,
        clock=lambda: next(_ticks),
    )
    try:
        c_wait.wait_run("abc123", poll_s=0.01, timeout_s=0.05)
        out["client_wait_run_timeout_raises"] = False
    except HarnessTransportError:
        out["client_wait_run_timeout_raises"] = True

    # ---- readiness + blocking drain ---------------------------------------
    ops_calls: list[tuple[str, str]] = []

    def _ops_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        ops_calls.append((method, url))
        if "/harness/version" in url:
            return (
                200,
                {"X-Fx1-Api-Version": "1"},
                b'{"api_version": "1", "fx1_version": "0.4.0"}',
            )
        if "/ready" in url:
            return 200, {}, b'{"ready": true, "inflight": 2}'
        return (
            200,
            {},
            _json_mod.dumps({"draining": True, "inflight": 0, "drained": True}).encode(),
        )

    c_ops = HarnessClient("http://harness.test", transport=_ops_transport)
    rd = c_ops.ready()
    out["client_ready_get"] = rd == {"ready": True, "inflight": 2}
    dr = c_ops.drain(wait_s=12.5)
    out["client_drain_wait_s_sent"] = any("wait_s=12.5" in u for _m, u in ops_calls)
    out["client_drain_reports_drained"] = dr["drained"] is True
    out["client_server_version"] = c_ops.server_version() == {
        "api_version": "1",
        "fx1_version": "0.4.0",
    }
    out["client_last_api_version"] = c_ops.last_api_version == "1"

    def _coded_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return (
            503,
            {},
            b'{"detail": "harness is draining - no new work", "code": "draining"}',
        )

    from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

    coded = HarnessClient("http://harness.test", transport=_coded_transport)
    try:
        coded.ready()
        out["client_error_code_carried"] = False
    except BackendNotConfiguredError as exc:
        out["client_error_code_carried"] = exc.code == "draining"
    return out


def parity_audit_bench() -> dict[str, Any]:
    r = parity_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "parity_audit",
        "schema": "parity_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "three-surface parity holds: SDK, HTTP API, and HarnessClient "
            "return byte-identical content, envelopes, chunk lists, error "
            "classes, verifier verdicts, registry, and health over the same "
            "injected backend; the in-flight cap fails saturated work with "
            "503+Retry-After while cheap routes respond, and releases "
            "cleanly. Flags: unknown backend names are KeyError in-process "
            "vs 422 literal rejection over the wire (request validation "
            "runs before resolution); empty batches are [] in-process vs "
            "422 over the wire."
            if ok
            else f"PARITY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
