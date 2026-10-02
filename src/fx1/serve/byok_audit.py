"""byok_audit — adversarial probes on the BYOK (bring-your-own-key) backend.

``OpenAICompatBackend`` is the bring-your-own-key path: the entire fx-1
eval/bench fleet runs against any user-declared OpenAI-compatible
endpoint while emitting the same sealed receipts and honesty gates.
That makes the backend's *contract* load-bearing — a silent fallback or
a malformed-payload pass would corrupt receipts for models fx-1 never
trained.

Pinned contract:

- Construction is fail-closed: each of ``FX1_BYOK_BASE_URL`` /
  ``FX1_BYOK_API_KEY`` / ``FX1_BYOK_MODEL`` missing raises
  BackendNotConfiguredError (the 503-class RuntimeError) naming the env
  vars; empty strings count as missing; a non-http(s) or
  hostless URL raises; a non-positive timeout raises ValueError.
- Explicit kwargs beat env (a caller's override is never shadowed by
  ambient process state).
- The request is pinned: ``temperature: 0.0`` (deterministic evals),
  Bearer auth carrying the key, ``Content-Type: application/json``, the
  model name from config, and the URL normalized to
  ``<base>/chat/completions`` (a complete ``.../chat/completions`` URL
  is accepted as-is; trailing slashes collapse).
- ``complete`` fails closed on transport errors (URLError →
  RuntimeError, never a fabricated completion) and on malformed
  payloads (missing ``choices[0].message.content``, non-string content).
- ``get_backend("byok")`` resolves and unknown kinds still fail closed.

Flagged warts (documented, not fixed):

- ``flag_http_allowed`` — ``http://`` bases are accepted because local
  vLLM/Ollama compat shims legitimately run on plain http; the key
  travels in clear over plaintext transports (caller-declared risk).
- ``flag_no_retry`` — a transient 5xx/timeout is an immediate
  RuntimeError; eval lanes must re-run rather than the backend
  double-submitting (a retried generation could differ under
  temperature>0 upstream configs, corrupting determinism).
- ``flag_streaming_absent`` — only unary completions are supported;
  ``stream`` is never requested so SSE-shaped endpoints that *require*
  streaming will fail closed via URLError/malformed payload.

Sealed ``byok_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import os
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["byok_audit", "byok_audit_bench"]

_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")


def byok_audit() -> dict[str, Any]:
    import json
    from unittest.mock import patch

    from fx1.serve.backends import (
        BYOK_API_KEY_ENV,
        BYOK_BASE_URL_ENV,
        BYOK_MODEL_ENV,
        OpenAICompatBackend,
        _chat_completions_url,
        get_backend,
    )

    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    saved = {name: os.environ.pop(name, None) for name in _ENVS}
    try:
        # ---- construction fail-closed -----------------------------
        out["missing_all_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        os.environ[BYOK_BASE_URL_ENV] = "https://probe.local/v1"
        out["missing_key_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        os.environ[BYOK_API_KEY_ENV] = "byok-probe-key"
        out["missing_model_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        os.environ[BYOK_MODEL_ENV] = "probe-model"
        # error message must name the missing vars, not leak values
        for name in _ENVS:
            os.environ.pop(name)
        try:
            OpenAICompatBackend()
            out["error_names_env"] = False
        except RuntimeError as exc:
            msg = str(exc)
            out["error_names_env"] = all(n in msg for n in _ENVS) and (
                "probe.local" not in msg and "byok-probe-key" not in msg
            )
        # empty string is missing
        os.environ[BYOK_BASE_URL_ENV] = ""
        os.environ[BYOK_API_KEY_ENV] = "k"
        os.environ[BYOK_MODEL_ENV] = "m"
        out["empty_base_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        # scheme / host validation
        os.environ[BYOK_BASE_URL_ENV] = "ftp://probe.local/v1"
        out["bad_scheme_raises"] = _raises(lambda: OpenAICompatBackend()) == "RuntimeError"
        os.environ[BYOK_BASE_URL_ENV] = "not-a-url"
        out["hostless_raises"] = _raises(lambda: OpenAICompatBackend()) == "RuntimeError"
        os.environ[BYOK_BASE_URL_ENV] = "https://probe.local/v1"
        out["bad_timeout_raises"] = (
            _raises(lambda: OpenAICompatBackend(timeout_s=0)) == "ValueError"
        )
        # kwargs beat env
        os.environ[BYOK_BASE_URL_ENV] = "https://env.local/v1"
        os.environ[BYOK_MODEL_ENV] = "env-model"
        kw = OpenAICompatBackend(base_url="https://kw.local/v1", model="kw-model")
        out["kwargs_beat_env"] = kw._model == "kw-model" and "kw.local" in kw._url
        os.environ[BYOK_BASE_URL_ENV] = "https://probe.local/v1"
        os.environ[BYOK_API_KEY_ENV] = "byok-probe-key"
        os.environ[BYOK_MODEL_ENV] = "probe-model"

        # ---- request pinning -------------------------------------
        captured: dict[str, Any] = {}

        class _Resp:
            def read(self) -> bytes:
                return json.dumps({"choices": [{"message": {"content": "probe-answer"}}]}).encode()

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        def fake_urlopen(req: Any, **kw: Any) -> _Resp:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            captured["auth"] = req.headers.get("Authorization")
            captured["ctype"] = req.headers.get("Content-type")
            captured["timeout"] = kw.get("timeout")
            return _Resp()

        import urllib.request  # noqa: PLC0415

        backend = OpenAICompatBackend()
        with patch.object(urllib.request, "urlopen", fake_urlopen):
            text = backend.complete([{"role": "user", "content": "hi"}])
        out["complete_roundtrip"] = text == "probe-answer"
        out["temperature_zero"] = captured["body"].get("temperature") == 0.0
        out["bearer_auth"] = captured["auth"] == "Bearer byok-probe-key"
        out["model_in_body"] = captured["body"].get("model") == "probe-model"
        out["json_content_type"] = captured["ctype"] == "application/json"
        out["url_normalized"] = captured["url"] == "https://probe.local/v1/chat/completions"
        out["timeout_wired"] = captured["timeout"] == 120

        # url normalization variants
        out["url_normalization"] = (
            _chat_completions_url("https://h/v1/") == "https://h/v1/chat/completions"
            and _chat_completions_url("https://h/v1/chat/completions")
            == "https://h/v1/chat/completions"
            and _chat_completions_url("http://localhost:8000")
            == "http://localhost:8000/chat/completions"
        )

        # ---- transport + payload fail-closed ----------------------
        def boom(req: Any, **kw: Any) -> _Resp:
            raise urllib.error.URLError("connection refused")

        import urllib.error  # noqa: PLC0415

        with patch.object(urllib.request, "urlopen", boom):
            out["transport_error_raises"] = _raises(lambda: backend.complete([])) == "RuntimeError"

        class _BadResp:
            def read(self) -> bytes:
                return json.dumps({"choices": [{"message": {}}]}).encode()

            def __enter__(self) -> _BadResp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        with patch.object(urllib.request, "urlopen", lambda req, **kw: _BadResp()):
            out["missing_content_raises"] = _raises(lambda: backend.complete([])) == "RuntimeError"

        class _NonStrResp:
            def read(self) -> bytes:
                return json.dumps({"choices": [{"message": {"content": 42}}]}).encode()

            def __enter__(self) -> _NonStrResp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        with patch.object(urllib.request, "urlopen", lambda req, **kw: _NonStrResp()):
            out["nonstr_content_raises"] = _raises(lambda: backend.complete([])) == "RuntimeError"

        # ---- flags ------------------------------------------------
        out["flag_http_allowed"] = True  # http:// bases accepted for local stacks
        out["flag_no_retry"] = True  # transport failure is terminal, not retried
        out["flag_streaming_absent"] = True  # unary only; stream never requested

        # ---- factory -----------------------------------------------
        out["factory_resolves_byok"] = isinstance(get_backend("byok"), OpenAICompatBackend)
        out["factory_unknown_closed"] = _raises(lambda: get_backend("nope")) == "KeyError"
    finally:
        for name, val in saved.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val

    return out


def byok_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under byok_audit.v1."""
    r = byok_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "byok_audit",
        "schema": "byok_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "BYOK backend holds: construction refuses missing "
            "credentials/URLs/models, kwargs beat env, the request pins "
            "temperature=0 + Bearer auth + normalized "
            "chat/completions URL, and malformed payloads or transport "
            "errors fail closed. Flags: http bases allowed for local "
            "stacks, no retries, unary only."
            if ok
            else f"BYOK AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
