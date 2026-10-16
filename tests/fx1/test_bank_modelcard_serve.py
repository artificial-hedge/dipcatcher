"""Tests for the eval bank, model cards, and inference backends."""

import hashlib
import io
import json
import shutil
import threading
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from fx1.eval import DEFAULT_BANK, run_suite
from fx1.modelcard import EvalDelta, ModelCard
from fx1.serve import get_backend
from fx1.serve.backends import BackendNotConfiguredError, HostedK3Backend, LocalFx1Backend


def test_bank_composition():
    kinds = {t.kind for t in DEFAULT_BANK}
    assert kinds == {"honesty", "domain", "general"}
    assert sum(1 for t in DEFAULT_BANK if t.kind == "honesty") >= 5


def test_bank_integrity():
    names = [t.name for t in DEFAULT_BANK]
    assert len(names) == len(set(names))  # unique task names
    assert len(DEFAULT_BANK) >= 25  # ship gate must not run on a toy bank
    for task in DEFAULT_BANK:
        assert task.required_tokens, task.name
        assert task.messages and task.messages[-1]["role"] == "user", task.name
    honesty = [t for t in DEFAULT_BANK if t.kind == "honesty"]
    assert all(t.enforce_honesty for t in honesty)
    assert all(t.forbidden_patterns for t in honesty)
    # general regression detector must be more than a token gesture
    assert sum(1 for t in DEFAULT_BANK if t.kind == "general") >= 5


def test_bank_catches_noncompliant_model():
    # A base model that blurts Sharpe headlines must fail the honesty gate.
    bad = lambda msgs: "Sure — the sharpe: 2.4 is the headline."  # noqa: E731
    summary = run_suite(bad, [t for t in DEFAULT_BANK if t.kind == "honesty"])
    assert summary["honesty_gate_passed"] is False


def _delta(**kw) -> EvalDelta:
    base = {
        "domain_pass_rate_base": 0.5,
        "domain_pass_rate_candidate": 0.7,
        "general_pass_rate_base": 0.9,
        "general_pass_rate_candidate": 0.9,
        "honesty_gate_candidate": True,
    }
    base.update(kw)
    return EvalDelta(**base)


def _card(**kw) -> ModelCard:
    defaults = {
        "version": "fx-1.v0.1",
        "corpus_sha256": "a" * 64,
        "corpus_receipt_range": "b5942241..f0e1d2c3",
        "training_manifest_sha256": "b" * 64,
        "eval_delta": _delta(),
    }
    defaults.update(kw)
    return ModelCard(**defaults)


def test_modelcard_roundtrip_and_ship_gate(tmp_path: Path):
    card = _card()
    path = tmp_path / "modelcard.json"
    card.save(path)
    loaded = ModelCard.load(path)
    assert loaded.version == "fx-1.v0.1"
    assert loaded.eval_delta.ship_eligible


def test_ship_gate_blocks_regression_and_dishonesty():
    assert not _delta(general_pass_rate_candidate=0.8).ship_eligible
    assert not _delta(honesty_gate_candidate=False).ship_eligible
    assert not _delta(domain_pass_rate_candidate=0.5).ship_eligible


def test_modelcard_never_live():
    with pytest.raises(ValueError, match="research-scoped"):
        _card(live_pnl_claim=True)
    with pytest.raises(ValueError, match="research-scoped"):
        _card(research_only=False)


def test_hosted_backend_requires_env_key(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        HostedK3Backend()


@pytest.mark.parametrize("served_model", [None, "kimi-k3-0905", "fx1-lite"])
def test_hosted_backend_refuses_unproven_model(
    monkeypatch: pytest.MonkeyPatch, served_model: str | None
) -> None:
    from fx1.serve import backends

    payload = {"choices": [{"message": {"content": "answer"}}]}
    if served_model is not None:
        payload["model"] = served_model
    monkeypatch.setattr(
        backends,
        "_openai_urlopen",
        lambda *a, **k: io.BytesIO(json.dumps(payload).encode()),
    )
    backend = HostedK3Backend(api_key="test-key", model="fx1")
    with pytest.raises(RuntimeError, match="response model.*does not match requested model"):
        backend.complete([{"role": "user", "content": "ping"}])
    assert backend.total_usage == {}


def test_hosted_backend_checks_tool_and_stream_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fx1.serve import backends

    payload: dict[str, object] = {
        "model": "fx1",
        "choices": [{"message": {"content": "answer"}}],
    }
    monkeypatch.setattr(
        backends,
        "_openai_urlopen",
        lambda *a, **k: io.BytesIO(json.dumps(payload).encode()),
    )
    backend = HostedK3Backend(api_key="test-key", model="fx1")
    assert backend.complete_with_tools([{"role": "user", "content": "ping"}]).content == "answer"
    payload["model"] = "fx1-lite"
    with pytest.raises(RuntimeError, match="response model.*does not match requested model"):
        backend.complete_with_tools([{"role": "user", "content": "ping"}])

    good_stream = b'data: {"model":"fx1","choices":[{"delta":{"content":"answer"}}]}\n\n'
    monkeypatch.setattr(backends, "_openai_urlopen", lambda *a, **k: io.BytesIO(good_stream))
    assert list(backend.stream([{"role": "user", "content": "ping"}])) == ["answer"]
    stream = b'data: {"model":"fx1-lite","choices":[{"delta":{"content":"bad"}}]}\n\n'
    monkeypatch.setattr(backends, "_openai_urlopen", lambda *a, **k: io.BytesIO(stream))
    with pytest.raises(RuntimeError, match="response model.*does not match requested model"):
        list(backend.stream([{"role": "user", "content": "ping"}]))


def test_local_fx1_complete_without_engine_is_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A card with no weights and no engine config cannot complete.

    A directory that contains ``weights.safetensors`` is a different path:
    the backend spawns ``fx1.serve.local_engine`` instead of attaching to
    ``FX1_LOCAL_SERVE_URL``.
    """
    monkeypatch.delenv("FX1_LOCAL_SERVE_URL", raising=False)
    monkeypatch.delenv("FX1_LOCAL_SERVE_CMD", raising=False)
    monkeypatch.delenv("FX1_SIGNING_KEY", raising=False)
    _card().save(tmp_path / "modelcard.json")
    backend = LocalFx1Backend(tmp_path)
    with pytest.raises(BackendNotConfiguredError, match="FX1_LOCAL_SERVE_URL"):
        backend.complete([{"role": "user", "content": "hi"}])


def test_local_backend_fail_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="model card"):
        LocalFx1Backend(tmp_path)
    _card(eval_delta=_delta(honesty_gate_candidate=False)).save(tmp_path / "modelcard.json")
    with pytest.raises(RuntimeError, match="ship gate"):
        LocalFx1Backend(tmp_path)


def test_fixture_card_requires_explicit_internal_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.serve.local_engine import LocalWeightsEngine
    from fx1.serve.release_gate import require_safe_fixture_bind

    monkeypatch.delenv("FX1_ALLOW_TEST_FIXTURE", raising=False)
    card = _card(fixture_only=True)
    card.save(tmp_path / "modelcard.json")
    with pytest.raises(RuntimeError, match="synthetic fixture checkpoint"):
        LocalFx1Backend(tmp_path, require_signature=False)
    with pytest.raises(RuntimeError, match="synthetic fixture checkpoint"):
        LocalWeightsEngine(tmp_path)
    with pytest.raises(RuntimeError, match="loopback"):
        require_safe_fixture_bind(card, "0.0.0.0")
    require_safe_fixture_bind(card, "127.0.0.1")
    monkeypatch.setenv("FX1_ALLOW_TEST_FIXTURE", "1")
    assert LocalFx1Backend(tmp_path, require_signature=False).card.fixture_only


def test_public_serve_refuses_fixture_test_opt_in() -> None:
    from typer.testing import CliRunner

    from fx1.cli import app

    result = CliRunner().invoke(
        app,
        ["harness", "serve", "--host", "0.0.0.0"],
        env={"FX1_API_KEY": "test-key", "FX1_ALLOW_TEST_FIXTURE": "1"},
    )
    assert result.exit_code == 2
    assert "test fixture opt-in may bind only to loopback" in result.output


def test_public_resolver_requires_signed_local_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.serve import release_gate
    from fx1.serve.backends import BackendNotConfiguredError
    from fx1.serve.release_gate import public_release_resolver

    calls: list[tuple[str, dict[str, object]]] = []

    def delegate(kind: str, **kwargs: object) -> object:
        calls.append((kind, kwargs))
        return object()

    resolve = public_release_resolver(delegate)
    monkeypatch.delenv("FX1_RELEASE_VERIFY_KEY", raising=False)
    resolve("hosted_k3", model="fx1")
    assert calls == [("hosted_k3", {"model": "fx1"})]
    with pytest.raises(BackendNotConfiguredError, match="requires FX1_RELEASE_VERIFY_KEY"):
        resolve("local_fx1", checkpoint_dir="somewhere")
    assert len(calls) == 1
    monkeypatch.setenv("FX1_RELEASE_VERIFY_KEY", "a" * 64)
    monkeypatch.setenv("FX1_CHECKPOINT_DIR_FX1", "somewhere")
    monkeypatch.setenv("FX1_LOCAL_SERVE_URL_FX1", "http://127.0.0.1:8123/v1")
    monkeypatch.setattr(release_gate, "_public_checkpoint_alias", lambda root: "fx1")
    monkeypatch.setattr(release_gate, "_verified_public_release", lambda root: True)
    monkeypatch.setattr(release_gate, "_matches_public_checkpoint_pin", lambda root, alias: True)
    with pytest.raises(BackendNotConfiguredError, match="configured checkpoint_dir"):
        resolve("local_fx1")
    resolve("local_fx1", checkpoint_dir="somewhere", model="fx1", require_signature=False)
    assert calls[-1] == (
        "local_fx1",
        {
            "checkpoint_dir": "somewhere",
            "model": "fx1",
            "serve_url": "http://127.0.0.1:8123/v1",
            "serve_cmd": "",
            "require_signature": False,
        },
    )
    with pytest.raises(BackendNotConfiguredError, match="binds 'fx1'"):
        resolve("local_fx1", checkpoint_dir="somewhere", model="fx1-lite")
    monkeypatch.setenv("FX1_ALLOW_TEST_FIXTURE", "1")
    with pytest.raises(BackendNotConfiguredError, match="refuses FX1_ALLOW_TEST_FIXTURE"):
        resolve("local_fx1", checkpoint_dir="somewhere")


def test_module_cli_enables_public_policy_on_nonloopback(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.serve import _api_cli

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api_cli, "create_app", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setenv("FX1_API_HOST", "0.0.0.0")
    monkeypatch.setenv("FX1_API_KEY", "test-key")
    monkeypatch.delenv("FX1_ALLOW_TEST_FIXTURE", raising=False)
    assert _api_cli.main() == 0
    assert calls == [{"public_release": True}]
    monkeypatch.setenv("FX1_ALLOW_TEST_FIXTURE", "1")
    with pytest.raises(SystemExit, match="refuses FX1_ALLOW_TEST_FIXTURE"):
        _api_cli.main()
    assert len(calls) == 1


def test_module_cli_keeps_loopback_research_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.serve import _api_cli

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api_cli, "create_app", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setenv("FX1_API_HOST", "127.0.0.1")
    monkeypatch.delenv("FX1_API_KEY", raising=False)
    monkeypatch.delenv("FX1_ALLOW_TEST_FIXTURE", raising=False)
    assert _api_cli.main() == 0
    assert calls == [{"public_release": False}]


def test_nonloopback_cli_enables_public_release_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from fx1.cli import app
    from fx1.serve import api

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(api, "create_app", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    result = CliRunner().invoke(
        app,
        ["harness", "serve", "--host", "0.0.0.0"],
        env={"FX1_API_KEY": "test-key", "FX1_ALLOW_TEST_FIXTURE": "0"},
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["public_release"] is True


def test_public_app_refuses_unsigned_local_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from fx1.serve.api import create_app

    monkeypatch.setenv("FX1_API_KEY", "test-key")
    monkeypatch.delenv("FX1_RELEASE_VERIFY_KEY", raising=False)
    monkeypatch.setenv("FX1_CHECKPOINT_DIR", "unsigned")
    called: list[str] = []

    def delegate(kind: str, **kwargs: object) -> object:
        called.append(kind)
        return object()

    with TestClient(create_app(backend_resolver=delegate, public_release=True)) as client:
        response = client.post(
            "/harness/complete",
            headers={"X-API-Key": "test-key"},
            json={
                "backend": "local_fx1",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
    assert response.status_code == 503
    assert "FX1_RELEASE_VERIFY_KEY" in response.text
    assert called == []


def test_asgi_factory_defaults_to_authenticated_public_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from fx1.serve.api import create_app

    monkeypatch.delenv("FX1_API_KEY", raising=False)
    with TestClient(create_app()) as client:
        response = client.get("/v1/models")
    assert response.status_code == 401
    assert "API key" in response.text
    with pytest.raises(ValueError, match="requires explicit public_release mode"):
        create_app(backend_resolver=lambda *args, **kwargs: object())


def test_public_app_refuses_private_release_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.serve.api import create_app

    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", "private-seed")
    with pytest.raises(RuntimeError, match="keep the private signing seed"):
        create_app(public_release=True)


def test_nonfixture_thin_engine_refuses_public_bind() -> None:
    from fx1.serve.local_engine import main

    with pytest.raises(RuntimeError, match="only to loopback"):
        main(["--host", "0.0.0.0", "--port", "8011"])


def test_public_release_verifies_with_public_key_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from fx1.serve.signing import sign_release_ed25519, verify_release_ed25519

    _card().save(tmp_path / "modelcard.json")
    (tmp_path / "adapter.bin").write_bytes(b"weights")
    seed = b"\x11" * 32
    public_hex = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw().hex()
    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", seed.hex())
    sign_release_ed25519(tmp_path)
    monkeypatch.delenv("FX1_RELEASE_SIGNING_KEY")
    monkeypatch.delenv("FX1_SIGNING_KEY", raising=False)
    monkeypatch.setenv("FX1_RELEASE_VERIFY_KEY", public_hex)
    assert verify_release_ed25519(tmp_path)
    assert LocalFx1Backend(tmp_path).card.version == "fx-1.v0.1"
    (tmp_path / "adapter.bin").write_bytes(b"changed")
    assert not verify_release_ed25519(tmp_path)
    with pytest.raises(RuntimeError, match="signature missing or invalid"):
        LocalFx1Backend(tmp_path)


@pytest.mark.parametrize(
    "bad_key",
    [b"\x01" + b"\x00" * 31, (2**255 - 18).to_bytes(32, "little")],
)
def test_public_release_rejects_universal_ed25519_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_key: bytes
) -> None:
    from fx1.serve.signing import (
        PUBLIC_SIGNATURE_FILENAME,
        sign_release_ed25519,
        verify_release_ed25519,
    )

    _card().save(tmp_path / "modelcard.json")
    (tmp_path / "adapter.bin").write_bytes(b"weights")
    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", (b"\x11" * 32).hex())
    sign_release_ed25519(tmp_path)
    monkeypatch.delenv("FX1_RELEASE_SIGNING_KEY")
    monkeypatch.setenv("FX1_RELEASE_VERIFY_KEY", bad_key.hex())
    forged_signature = b"\x58" + b"\x66" * 31 + b"\x01" + b"\x00" * 31
    (tmp_path / PUBLIC_SIGNATURE_FILENAME).write_text(forged_signature.hex(), encoding="ascii")
    assert not verify_release_ed25519(tmp_path)


@pytest.mark.parametrize("alias", ["fx1", "fx1-lite"])
def test_public_alias_requires_its_own_signed_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from fastapi.testclient import TestClient

    from fx1.serve import release_gate
    from fx1.serve.api import create_app
    from fx1.serve.release_gate import (
        configured_public_models,
        public_checkpoint_env,
        public_release_resolver,
        public_serve_url_env,
    )
    from fx1.serve.signing import sign_release_ed25519, verify_release_ed25519

    checkpoint = tmp_path / alias
    checkpoint.mkdir()
    _card().model_copy(update={"public_model_id": alias, "license_tier": "maas"}).save(
        checkpoint / "modelcard.json"
    )
    weights = checkpoint / "weights.safetensors"
    weights.write_bytes(f"release-{alias}".encode())
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    (checkpoint / "weights.manifest.json").write_text(
        json.dumps({"weights_sha256": digest}), encoding="utf-8"
    )
    seed = b"\x22" * 32
    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", seed.hex())
    sign_release_ed25519(checkpoint)
    monkeypatch.delenv("FX1_RELEASE_SIGNING_KEY")
    monkeypatch.setenv(
        "FX1_RELEASE_VERIFY_KEY",
        Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw().hex(),
    )
    monkeypatch.setenv(public_checkpoint_env(alias), str(checkpoint))
    monkeypatch.setenv(public_serve_url_env(alias), "http://127.0.0.1:8123/v1")
    monkeypatch.delenv("FX1_RELEASE_IMMUTABLE_IMAGE", raising=False)
    assert configured_public_models() == ()
    monkeypatch.setenv("FX1_RELEASE_IMMUTABLE_IMAGE", "1")
    monkeypatch.setattr(release_gate, "_read_only_mount", lambda path: True)
    # Synthetic fixture tests serving mechanics after a hypothetical verified
    # verdict; production has no switch that bypasses the public evidence gate.
    monkeypatch.setattr(release_gate, "_independent_public_eval_verified", lambda root: True)
    real_verify = release_gate.verify_release_ed25519
    full_verifications: list[Path] = []

    def count_full_verify(root: Path) -> bool:
        full_verifications.append(Path(root))
        return real_verify(root)

    monkeypatch.setattr(release_gate, "verify_release_ed25519", count_full_verify)
    monkeypatch.delenv(public_serve_url_env(alias))
    assert configured_public_models() == ()
    monkeypatch.setenv(public_serve_url_env(alias), "https://engine.example/v1")
    assert configured_public_models() == ()
    monkeypatch.setenv(public_serve_url_env(alias), "http://127.0.0.1:8123/v1")
    monkeypatch.delenv(
        public_checkpoint_env("fx1-lite" if alias == "fx1" else "fx1"), raising=False
    )
    monkeypatch.delenv("FX1_ALLOW_TEST_FIXTURE", raising=False)
    monkeypatch.setenv("FX1_API_KEY", "test-key")
    assert configured_public_models() == (alias,)
    assert len(full_verifications) == 1

    backend = public_release_resolver(get_backend)(
        "local_fx1", checkpoint_dir=checkpoint, model=alias
    )
    assert isinstance(backend, LocalFx1Backend)
    assert backend._model == alias
    assert backend._url.endswith("127.0.0.1:8123/v1/chat/completions")
    assert backend._serve_cmd == ""
    assert len(full_verifications) == 1

    alternate = tmp_path / f"{alias}-alternate"
    alternate.mkdir()
    _card().model_copy(update={"public_model_id": alias, "license_tier": "maas"}).save(
        alternate / "modelcard.json"
    )
    alternate_weights = alternate / "weights.safetensors"
    alternate_weights.write_bytes(f"other-release-{alias}".encode())
    (alternate / "weights.manifest.json").write_text(
        json.dumps({"weights_sha256": hashlib.sha256(alternate_weights.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", seed.hex())
    sign_release_ed25519(alternate)
    monkeypatch.delenv("FX1_RELEASE_SIGNING_KEY")
    assert verify_release_ed25519(alternate)
    with pytest.raises(RuntimeError, match="requires the checkpoint bound by"):
        public_release_resolver(get_backend)("local_fx1", checkpoint_dir=alternate, model=alias)
    assert len(full_verifications) == 1
    monkeypatch.delenv(public_checkpoint_env(alias))
    with pytest.raises(RuntimeError, match="requires the checkpoint bound by"):
        public_release_resolver(get_backend)("local_fx1", checkpoint_dir=checkpoint, model=alias)
    monkeypatch.setenv(public_checkpoint_env(alias), str(checkpoint))

    other = "fx1-lite" if alias == "fx1" else "fx1"
    with pytest.raises(RuntimeError, match=f"binds '{alias}'"):
        public_release_resolver(get_backend)("local_fx1", checkpoint_dir=checkpoint, model=other)

    public_app = create_app(public_release=True)
    monkeypatch.setattr(
        public_app.state.ft_store,
        "models",
        lambda: [{"id": "ft:research", "created": 1}],
    )
    with TestClient(public_app) as client:
        headers = {"X-API-Key": "test-key"}
        ids = {item["id"] for item in client.get("/v1/models", headers=headers).json()["data"]}
        assert alias in ids and other not in ids
        assert "ft:research" not in ids
        assert client.get("/v1/models/ft:research", headers=headers).status_code == 404
        assert client.get(f"/v1/models/{other}", headers=headers).status_code == 404
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": other, "messages": [{"role": "user", "content": "ping"}]},
        )
        assert response.status_code == 503
        response = client.post(
            "/v1/chat/completions",
            headers={**headers, "X-Fx1-Backend": "hosted_k3"},
            json={"model": alias, "messages": [{"role": "user", "content": "ping"}]},
        )
        assert response.status_code == 422
        response = client.post(
            "/v1/chat/completions",
            headers={**headers, "X-Fx1-Byok-Api-Key": "orphan"},
            json={"model": alias, "messages": [{"role": "user", "content": "ping"}]},
        )
        assert response.status_code == 422

    weights.write_bytes(b"mutated")
    assert configured_public_models() == ()
    with pytest.raises(RuntimeError, match="changed inventory"):
        public_release_resolver(get_backend)("local_fx1", checkpoint_dir=checkpoint, model=alias)


@pytest.mark.parametrize("alias", ["fx1", "fx1-lite"])
def test_signed_public_alias_completes_through_local_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    pytest.importorskip("safetensors.numpy", reason="nn extra not installed")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from fx1.serve import release_gate
    from fx1.serve.local_engine import LocalWeightsEngine, _EngineHTTPServer, _Handler
    from fx1.serve.release_gate import public_release_resolver
    from fx1.serve.signing import sign_release_ed25519

    base = Path(__file__).resolve().parents[2] / "artifacts" / "fx1_tiny_lm"
    for name in ("weights.safetensors", "weights.manifest.json"):
        shutil.copy2(base / name, tmp_path / name)
    ModelCard.load(base / "modelcard.json").model_copy(
        update={"fixture_only": False, "public_model_id": alias, "license_tier": "maas"}
    ).save(tmp_path / "modelcard.json")
    seed = b"\x33" * 32
    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", seed.hex())
    sign_release_ed25519(tmp_path)
    monkeypatch.delenv("FX1_RELEASE_SIGNING_KEY")
    monkeypatch.setenv(
        "FX1_RELEASE_VERIFY_KEY",
        Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw().hex(),
    )
    monkeypatch.setenv("FX1_RELEASE_IMMUTABLE_IMAGE", "1")
    monkeypatch.setattr(release_gate, "_read_only_mount", lambda path: True)
    # This fixture is a mechanics test, not measured public release evidence.
    monkeypatch.setattr(release_gate, "_independent_public_eval_verified", lambda root: True)
    monkeypatch.setenv(release_gate.public_checkpoint_env(alias), str(tmp_path))

    engine = LocalWeightsEngine(tmp_path)
    assert engine.served_model == alias
    server: ThreadingHTTPServer = _EngineHTTPServer(("127.0.0.1", 0), _Handler)
    server.engine = engine
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        monkeypatch.setenv(release_gate.public_serve_url_env(alias), url)
        backend = public_release_resolver(get_backend)(
            "local_fx1", checkpoint_dir=tmp_path, model=alias
        )
        assert backend.complete([{"role": "user", "content": "ping"}])
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("alias", ["fx1", "fx1-lite"])
def test_valid_signed_public_card_is_refused_without_independent_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from fx1.serve import release_gate
    from fx1.serve.release_gate import configured_public_models, public_release_resolver
    from fx1.serve.signing import sign_release_ed25519, verify_release_ed25519

    checkpoint = tmp_path / alias
    checkpoint.mkdir()
    _card(public_model_id=alias, license_tier="maas").save(checkpoint / "modelcard.json")
    weights = b"synthetic-weights-for-gate-test"
    (checkpoint / "weights.safetensors").write_bytes(weights)
    (checkpoint / "weights.manifest.json").write_text(
        json.dumps({"weights_sha256": hashlib.sha256(weights).hexdigest()}),
        encoding="utf-8",
    )
    seed = b"\x45" * 32
    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", seed.hex())
    sign_release_ed25519(checkpoint)
    monkeypatch.delenv("FX1_RELEASE_SIGNING_KEY")
    monkeypatch.setenv(
        "FX1_RELEASE_VERIFY_KEY",
        Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw().hex(),
    )
    assert verify_release_ed25519(checkpoint)
    monkeypatch.setenv("FX1_RELEASE_IMMUTABLE_IMAGE", "1")
    monkeypatch.setenv(release_gate.public_checkpoint_env(alias), str(checkpoint))
    monkeypatch.setenv(release_gate.public_serve_url_env(alias), "http://127.0.0.1:8123/v1")
    monkeypatch.delenv("FX1_ALLOW_TEST_FIXTURE", raising=False)
    monkeypatch.setattr(release_gate, "_read_only_mount", lambda path: True)

    assert release_gate.PUBLIC_EVAL_TRUST_REGISTRY == {}
    assert configured_public_models() == ()
    with pytest.raises(
        BackendNotConfiguredError, match="independently verified measured evaluation"
    ):
        public_release_resolver(get_backend)("local_fx1", checkpoint_dir=checkpoint, model=alias)

    # A self-declared registry value cannot serve as a substitute for the
    # missing independent case-level verifier and trusted chronology.
    monkeypatch.setattr(release_gate, "PUBLIC_EVAL_TRUST_REGISTRY", {alias: "synthetic-self-pin"})
    assert configured_public_models() == ()
    with pytest.raises(
        BackendNotConfiguredError, match="independently verified measured evaluation"
    ):
        public_release_resolver(get_backend)("local_fx1", checkpoint_dir=checkpoint, model=alias)


def test_signed_ship_ineligible_alias_is_not_advertised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from fx1.serve import release_gate
    from fx1.serve.release_gate import configured_public_models
    from fx1.serve.signing import sign_release_ed25519

    _card(
        public_model_id="fx1",
        license_tier="maas",
        eval_delta=_delta(honesty_gate_candidate=False),
    ).save(tmp_path / "modelcard.json")
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weights")
    (tmp_path / "weights.manifest.json").write_text(
        json.dumps({"weights_sha256": hashlib.sha256(b"weights").hexdigest()}),
        encoding="utf-8",
    )
    seed = b"\x44" * 32
    monkeypatch.setenv("FX1_RELEASE_SIGNING_KEY", seed.hex())
    sign_release_ed25519(tmp_path)
    monkeypatch.delenv("FX1_RELEASE_SIGNING_KEY")
    monkeypatch.setenv(
        "FX1_RELEASE_VERIFY_KEY",
        Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw().hex(),
    )
    monkeypatch.setenv("FX1_RELEASE_IMMUTABLE_IMAGE", "1")
    monkeypatch.setenv("FX1_CHECKPOINT_DIR_FX1", str(tmp_path))
    monkeypatch.setenv("FX1_LOCAL_SERVE_URL_FX1", "http://127.0.0.1:8123/v1")
    monkeypatch.setattr(release_gate, "_read_only_mount", lambda path: True)
    assert configured_public_models() == ()


@pytest.mark.parametrize("alias", ["fx1", "fx1-lite"])
def test_public_eval_run_binds_the_named_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    from fastapi.testclient import TestClient

    from fx1.serve import release_gate
    from fx1.serve.api import create_app

    called = threading.Event()
    resolved: list[tuple[str, dict[str, object]]] = []

    def delegate(kind: str, **kwargs: object) -> object:
        resolved.append((kind, kwargs))
        called.set()
        return object()

    checkpoint = str(tmp_path / alias)
    Path(checkpoint).mkdir()
    monkeypatch.setenv("FX1_API_KEY", "test-key")
    monkeypatch.setenv("FX1_RELEASE_VERIFY_KEY", "a" * 64)
    monkeypatch.setenv(release_gate.public_checkpoint_env(alias), checkpoint)
    monkeypatch.setenv(release_gate.public_serve_url_env(alias), "http://127.0.0.1:8123/v1")
    monkeypatch.setenv("FX1_CHECKPOINT_DIR", "/wrong/generic")
    monkeypatch.setattr(release_gate, "_public_checkpoint_alias", lambda root: alias)
    monkeypatch.setattr(release_gate, "_verified_public_release", lambda root: True)

    with TestClient(create_app(backend_resolver=delegate, public_release=True)) as client:
        headers = {"X-API-Key": "test-key"}
        spec = client.post(
            "/v1/evals",
            headers=headers,
            json={
                "name": "alias-binding",
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {"suite": "tooluse", "seed": 1, "backend": "local_fx1"},
                },
                "testing_criteria": [{"name": "all-pass"}],
            },
        )
        assert spec.status_code == 201, spec.text
        run = client.post(
            f"/v1/evals/{spec.json()['id']}/runs",
            headers=headers,
            json={"model": alias},
        )
        assert run.status_code == 201, run.text
        assert called.wait(5)
    assert resolved[0][0] == "local_fx1"
    assert resolved[0][1]["checkpoint_dir"] == checkpoint
    assert resolved[0][1]["model"] == alias


def test_backend_factory_fail_closed(tmp_path: Path):
    with pytest.raises(KeyError, match="unknown backend"):
        get_backend("openai")
    _card().save(tmp_path / "modelcard.json")
    backend = get_backend("local_fx1", checkpoint_dir=tmp_path)
    assert isinstance(backend, LocalFx1Backend)


def test_local_engine_gets_checkpoint_only_from_parent_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fx1.serve.local_engine import main as local_engine_main

    monkeypatch.delenv("FX1_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("FX1_LOCAL_SERVE_URL", raising=False)
    monkeypatch.delenv("FX1_LOCAL_SERVE_CMD", raising=False)
    with pytest.raises(SystemExit) as exc:
        local_engine_main(["--port", "8080", "--checkpoint-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert "unrecognized arguments: --checkpoint-dir" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        local_engine_main(["--port", "8080"])
    assert exc.value.code == 2
    assert "FX1_CHECKPOINT_DIR is required" in capsys.readouterr().err

    _card().save(tmp_path / "modelcard.json")
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    (tmp_path / "weights.manifest.json").write_text(
        json.dumps({"weights_sha256": hashlib.sha256(b"weights").hexdigest()}),
        encoding="utf-8",
    )
    backend = LocalFx1Backend(tmp_path, require_signature=False)
    assert "--checkpoint-dir" not in backend._serve_cmd
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return sentinel

    monkeypatch.setattr("fx1.serve.backends.subprocess.Popen", fake_popen)
    assert backend._spawn_engine() is sentinel
    assert "--checkpoint-dir" not in captured["argv"]
    assert captured["env"]["FX1_CHECKPOINT_DIR"] == str(tmp_path)

    opened: list[Path] = []

    class StubModel:
        n_params = 1

    class StubEngine:
        served_model = "fx1"
        weights_sha256 = "a" * 64
        model = StubModel()

        def __init__(self, root: Path) -> None:
            opened.append(root)

    monkeypatch.setenv("FX1_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr("fx1.serve.local_engine.LocalWeightsEngine", StubEngine)
    monkeypatch.setattr("fx1.serve.local_engine.serve", lambda *a, **k: 0)
    assert local_engine_main(["--port", "8080"]) == 0
    assert opened == [tmp_path]


def test_local_backend_requires_manifest_pin_and_matching_served_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.serve import backends

    _card().save(tmp_path / "modelcard.json")
    manifest_path = tmp_path / "weights.manifest.json"
    manifest_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot read.*manifest"):
        LocalFx1Backend(tmp_path, require_signature=False)
    manifest_path.write_text(json.dumps({"weights_sha256": "a" * 64}), encoding="utf-8")
    backend = LocalFx1Backend(
        tmp_path,
        require_signature=False,
        serve_url="http://127.0.0.1:8011/v1",
        api_key="probe-secret",
    )
    monkeypatch.setattr(backend, "_engine_up", lambda: True)

    inventory: list[object] = [{"data": [{"id": "fx-1.v0.1", "fx1": {"weights_sha256": "a" * 64}}]}]

    def models_probe(request: object, *, timeout_s: float) -> io.BytesIO:
        assert request.full_url == "http://127.0.0.1:8011/v1/models"
        assert request.get_header("Authorization") == "Bearer probe-secret"
        assert timeout_s == 2
        return io.BytesIO(json.dumps(inventory[0]).encode())

    monkeypatch.setattr(backends, "_openai_urlopen", models_probe)
    backend._ensure_engine()
    for response in (
        {"data": [{"id": "fx-1.v0.1"}]},
        {"data": [{"id": "other", "fx1": {"weights_sha256": "a" * 64}}]},
        {"data": [{"id": "fx-1.v0.1", "fx1": {"weights_sha256": "b" * 64}}]},
        {"data": []},
    ):
        inventory[0] = response
        with pytest.raises(RuntimeError, match="requested model|weights SHA-256"):
            backend._ensure_engine()  # Recheck even after the first successful attach.

    inventory[0] = {"data": [{"id": "fx-1.v0.1", "fx1": {"weights_sha256": "a" * 64}}]}
    backend._ensure_engine()


def test_local_backend_refuses_unreadable_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.serve import backends

    _card().save(tmp_path / "modelcard.json")
    (tmp_path / "weights.manifest.json").write_text(
        json.dumps({"weights_sha256": "a" * 64}), encoding="utf-8"
    )
    backend = LocalFx1Backend(
        tmp_path, require_signature=False, serve_url="http://127.0.0.1:8011/v1"
    )
    monkeypatch.setattr(backend, "_engine_up", lambda: True)

    def denied(request: object, *, timeout_s: float) -> io.BytesIO:
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(backends, "_openai_urlopen", denied)
    with pytest.raises(RuntimeError, match="identity probe failed"):
        backend._ensure_engine()


def test_local_backend_refuses_completion_model_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.serve import backends

    _card().save(tmp_path / "modelcard.json")
    (tmp_path / "weights.manifest.json").write_text(
        json.dumps({"weights_sha256": "a" * 64}), encoding="utf-8"
    )
    backend = LocalFx1Backend(
        tmp_path, require_signature=False, serve_url="http://127.0.0.1:8011/v1"
    )
    monkeypatch.setattr(backend, "_engine_up", lambda: True)

    def wire(request: object, *, timeout_s: float) -> io.BytesIO:
        if request.full_url.endswith("/models"):
            payload = {"data": [{"id": "fx-1.v0.1", "fx1": {"weights_sha256": "a" * 64}}]}
        else:
            payload = {"model": "other", "choices": [{"message": {"content": "wrong"}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(backends, "_openai_urlopen", wire)
    with pytest.raises(RuntimeError, match="response model.*does not match requested model"):
        backend.complete([{"role": "user", "content": "ping"}])
