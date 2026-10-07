"""SYNTHETIC adversarial probes for fx1.serve.backends — transport seams,
credential hygiene, error-envelope honesty, and env-scope discipline."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

from fx1.modelcard import EvalDelta, ModelCard
from fx1.serve import backends
from fx1.serve.backends import (
    BackendNotConfiguredError,
    HostedK3Backend,
    LocalFx1Backend,
    OpenAICompatBackend,
    SamplingParams,
)

_MESSAGES = [{"role": "user", "content": "SYNTHETIC probe"}]


def _delta() -> EvalDelta:
    return EvalDelta(
        domain_pass_rate_base=0.5,
        domain_pass_rate_candidate=0.7,
        general_pass_rate_base=0.9,
        general_pass_rate_candidate=0.9,
        honesty_gate_candidate=True,
    )


def _checkpoint(root: Path) -> Path:
    ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b5942241..f0e1d2c3",
        training_manifest_sha256="b" * 64,
        eval_delta=_delta(),
    ).save(root / "modelcard.json")
    return root


def _urlopen_returning(payload: Any):
    """A ``_openai_urlopen`` stand-in that yields one canned JSON body."""

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(payload).encode()

        def close(self) -> None:
            pass

    @contextlib.contextmanager
    def fake_open(request: Any, *, timeout_s: float):
        del request, timeout_s
        yield _Resp()

    return fake_open


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 7}}]},
        ["not", "a", "dict"],
    ],
)
def test_hosted_k3_malformed_payloads_stay_inside_the_error_envelope(
    monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    """Every transport/parse failure this route can see must surface as the
    documented ``RuntimeError`` envelope — a raw ``KeyError``/``IndexError``/
    ``TypeError`` on the response shape is an envelope leak."""
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setattr(backends, "_openai_urlopen", _urlopen_returning(payload))
    backend = HostedK3Backend(api_key="SYNTHETIC-key")
    with pytest.raises(RuntimeError, match="malformed hosted_k3 completion payload"):
        backend.complete(list(_MESSAGES))


def test_hosted_k3_well_formed_payload_still_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setattr(
        backends,
        "_openai_urlopen",
        _urlopen_returning(
            {"choices": [{"message": {"content": "SYNTHETIC out"}}], "usage": {"total_tokens": 3}}
        ),
    )
    backend = HostedK3Backend(api_key="SYNTHETIC-key")
    assert backend.complete(list(_MESSAGES)) == "SYNTHETIC out"
    assert backend.last_usage == {"total_tokens": 3}


@pytest.mark.parametrize("surface", ["complete", "complete_with_tools", "stream", "count_tokens"])
def test_local_fx1_unconfigured_surfaces_refuse_before_any_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, surface: str
) -> None:
    """With no ``serve_url``/``serve_cmd`` configured every completion
    surface must raise ``BackendNotConfiguredError`` — none may spawn a
    process or touch the wire."""
    monkeypatch.delenv(backends.LOCAL_SERVE_URL_ENV, raising=False)
    monkeypatch.delenv(backends.LOCAL_SERVE_CMD_ENV, raising=False)
    spawned: list[Any] = []
    monkeypatch.setattr(
        backends.subprocess, "Popen", lambda *a, **k: spawned.append((a, k)) or None
    )
    monkeypatch.setattr(
        backends, "_openai_chat_stream", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    monkeypatch.setattr(
        backends, "_openai_chat_complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    monkeypatch.setattr(
        backends, "_openai_tokenize_count", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    backend = LocalFx1Backend(_checkpoint(tmp_path), serve_cmd="")
    if surface == "stream":
        call = lambda: list(backend.stream(list(_MESSAGES)))  # noqa: E731
    elif surface == "complete_with_tools":
        call = lambda: backend.complete_with_tools(list(_MESSAGES))  # noqa: E731
    elif surface == "count_tokens":
        call = lambda: backend.count_tokens(list(_MESSAGES))  # noqa: E731
    else:
        call = lambda: backend.complete(list(_MESSAGES))  # noqa: E731
    with pytest.raises(BackendNotConfiguredError):
        call()
    assert spawned == []


def test_local_fx1_stream_spawn_template_also_refuses_when_unattached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A spawn template does not make an unconfigured attach URL valid:
    ``stream`` must refuse before ``_ensure_engine`` can spawn."""
    monkeypatch.delenv(backends.LOCAL_SERVE_URL_ENV, raising=False)
    monkeypatch.delenv(backends.LOCAL_SERVE_CMD_ENV, raising=False)
    spawned: list[Any] = []
    monkeypatch.setattr(
        backends.subprocess, "Popen", lambda *a, **k: spawned.append((a, k)) or None
    )
    backend = LocalFx1Backend(_checkpoint(tmp_path), serve_cmd="spawn-me")
    with pytest.raises(BackendNotConfiguredError):
        list(backend.stream(list(_MESSAGES)))
    assert spawned == []


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "YeS"])
def test_byok_private_optin_accepts_only_affirmative_tokens(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, raw)
    assert backends._byok_private_networks_allowed() is True


@pytest.mark.parametrize("raw", ["", "0", "no", "on", "2", "truee", " yesno"])
def test_byok_private_optin_rejects_lookalike_tokens(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, raw)
    assert backends._byok_private_networks_allowed() is False


def test_byok_destination_policy_snapshots_env_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in is read at request-stamp time and never written back —
    toggling the env between calls flips the stamped policy only."""
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request("https://provider.invalid/v1/chat/completions")
    monkeypatch.delenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, raising=False)
    backends._apply_byok_destination_policy(req)
    assert req._fx1_byok_public_only is True
    assert req._fx1_byok_allow_private is False
    monkeypatch.setenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, "1")
    backends._apply_byok_destination_policy(req)
    assert req._fx1_byok_public_only is False
    assert req._fx1_byok_allow_private is True


def test_byok_request_path_never_mutates_the_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in must not leak past its scope: a complete() under the
    opt-in leaves ``os.environ`` byte-identical."""
    monkeypatch.setenv(backends.BYOK_ALLOW_PRIVATE_NETWORKS_ENV, "1")
    monkeypatch.setenv(
        backends.BYOK_API_KEY_ENV,
        "fx1k_synthetic",  # gitleaks:allow — probe literal
    )
    monkeypatch.setenv(backends.BYOK_BASE_URL_ENV, "https://provider.invalid/v1")
    monkeypatch.setenv(backends.BYOK_MODEL_ENV, "SYNTHETIC-model")
    monkeypatch.setattr(
        backends,
        "_openai_urlopen",
        _urlopen_returning({"choices": [{"message": {"content": "ok"}}]}),
    )
    import os  # noqa: PLC0415

    before = dict(os.environ)
    OpenAICompatBackend().complete(list(_MESSAGES))
    assert dict(os.environ) == before


def test_byok_missing_config_names_vars_not_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The not-configured error names the missing env vars — it must never
    echo a supplied credential back."""
    monkeypatch.delenv(backends.BYOK_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(backends.BYOK_MODEL_ENV, raising=False)
    monkeypatch.setenv(
        backends.BYOK_API_KEY_ENV,
        "fx1k_synthetic_secret",  # gitleaks:allow — probe literal
    )
    with pytest.raises(BackendNotConfiguredError) as denied:
        OpenAICompatBackend()
    text = str(denied.value)
    assert backends.BYOK_BASE_URL_ENV in text
    assert backends.BYOK_MODEL_ENV in text
    assert "fx1k_synthetic_secret" not in text


@pytest.mark.parametrize(
    "url, fragment",
    [
        ("ftp://provider.invalid/v1", "http(s)"),
        ("https://u:fx1k_secret_pw@provider.invalid/v1", "userinfo"),
        (
            "https://provider.invalid/v1?api_key=fx1k_secret_pw",  # gitleaks:allow — probe literal
            "query",
        ),
        ("https://provider.invalid/v1#fx1k_secret_pw", "fragment"),
        ("https://provider.invalid:notaport/v1", "port"),
        ("https:///v1", "http(s)"),
    ],
)
def test_byok_url_problem_wording_never_echoes_the_url(url: str, fragment: str) -> None:
    """``byok_base_url_problem`` returns reason wording only — a pasted
    secret inside the URL must not ride the error text."""
    problem = backends.byok_base_url_problem(url)
    assert problem is not None
    assert "fx1k_secret_pw" not in problem
    assert fragment in problem


def test_refuse_redirects_drops_every_redirect_code() -> None:
    """The redirect handler refuses each redirect status — a credentialed
    request is never replayed to a changed origin."""
    import urllib.request  # noqa: PLC0415

    handler = backends._RefuseRedirects()
    request = urllib.request.Request("https://provider.invalid/v1/chat/completions")
    for code in (301, 302, 303, 307, 308):
        assert (
            handler.redirect_request(
                request, None, code, "redirect", {}, "https://attacker.invalid/x"
            )
            is None
        )


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"prompt_tokens": 1, "total_tokens": 3}, {"prompt_tokens": 1, "total_tokens": 3}),
        ({"prompt_tokens": True, "total_tokens": 3}, {"total_tokens": 3}),
        ({"prompt_tokens": 1.5, "total_tokens": 3}, {"total_tokens": 3}),
        ({"prompt_tokens": "9", "total_tokens": 3}, {"total_tokens": 3}),
        ({"total_tokens": False}, None),
        ({"total_tokens": float("nan")}, None),
        ({"total_tokens": float("inf")}, None),
        ("not-a-dict", None),
        (None, None),
    ],
)
def test_extract_usage_sieves_unbillable_values(usage: Any, expected: Any) -> None:
    assert backends._extract_usage({"usage": usage}) == expected


def test_extract_usage_ignores_non_dict_payload() -> None:
    assert backends._extract_usage(["usage"]) is None
    assert backends._extract_usage("usage") is None


def test_sampling_params_default_is_deterministic() -> None:
    """temperature defaults to 0.0 — eval runs must be reproducible."""
    assert SamplingParams().body_fields()["temperature"] == 0.0


def test_openai_sibling_url_edges() -> None:
    chat = "https://provider.invalid/v1/chat/completions"
    assert backends._openai_sibling_url(chat, "tokenize").endswith("/v1/tokenize")
    assert backends._openai_sibling_url(chat + "/", "tokenize").endswith("/v1/tokenize")
    already = "https://provider.invalid/v1/tokenize"
    assert backends._openai_sibling_url(already, "tokenize") == already


def test_chat_completions_url_edges() -> None:
    assert backends._chat_completions_url("https://h/v1").endswith("/v1/chat/completions")
    assert backends._chat_completions_url("https://h/v1/").endswith("/v1/chat/completions")
    already = "https://h/v1/chat/completions"
    assert backends._chat_completions_url(already) == already
