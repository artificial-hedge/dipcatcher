"""Inference backends for fx-1.

Two backends, one interface:
- ``HostedK3Backend`` — Moonshot's hosted K3 API. Serves as fx-1's teacher
  (tool-use trace generation) and as the day-one serving path.
- ``LocalFx1Backend`` — a fine-tuned fx-1 checkpoint. Fail-closed until a
  checkpoint with a valid model card exists on disk.

Credentials come from environment variables only — never hardcoded.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol, cast

from fx1.modelcard import ModelCard

MOONSHOT_API_URL = "https://api.moonshot.ai/v1/chat/completions"

BYOK_BASE_URL_ENV = "FX1_BYOK_BASE_URL"
BYOK_API_KEY_ENV = "FX1_BYOK_API_KEY"
BYOK_MODEL_ENV = "FX1_BYOK_MODEL"


def _chat_completions_url(base_url: str) -> str:
    """Normalize a BYOK base to the chat-completions route.

    Accepts ``https://host/v1``, ``https://host/v1/``, or an already-
    complete ``.../chat/completions`` URL.
    """
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class InferenceBackend(Protocol):
    """Chat-completion interface shared by all fx-1 backends."""

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class HostedK3Backend:
    """Hosted Kimi K3 via the Moonshot API (stdlib HTTP; no new deps)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "kimi-k3",
        api_url: str = MOONSHOT_API_URL,
    ) -> None:
        self._api_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("MOONSHOT_API_KEY is not set; fx-1 never hardcodes credentials")
        self._model = model
        self._api_url = api_url

    def complete(self, messages: list[dict[str, str]]) -> str:
        # temperature pinned to 0 — eval/teacher runs must be deterministic;
        # unpinned sampling makes eval results unreproducible across replays.
        body = json.dumps({"model": self._model, "messages": messages, "temperature": 0.0}).encode()
        request = urllib.request.Request(
            self._api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 — pinned Moonshot API URL  # nosec B310
            payload = json.loads(response.read().decode())
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError(
                f"malformed completion payload: content is {type(content).__name__}, not str"
            )
        return content


class OpenAICompatBackend:
    """BYOK — any OpenAI-compatible chat-completions endpoint.

    Lets the whole fx-1 eval/bench fleet run against user-supplied
    providers (vLLM, SGLang, OpenRouter, Azure OpenAI, Ollama's compat
    shim, ...): every lane then emits the same sealed receipts and
    honesty gates regardless of whose weights sit behind the socket.

    Credentials come from ``FX1_BYOK_API_KEY`` (or the explicit kwarg)
    — never argv and never committed config. The endpoint is
    ``FX1_BYOK_BASE_URL`` + ``/chat/completions`` and the model name is
    ``FX1_BYOK_MODEL``. Explicit kwargs win over env. Fail-closed: a
    missing credential or a malformed payload raises, nothing is
    fabricated.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout_s: int = 120,
    ) -> None:
        url = base_url if base_url is not None else os.environ.get(BYOK_BASE_URL_ENV, "")
        key = api_key if api_key is not None else os.environ.get(BYOK_API_KEY_ENV, "")
        mdl = model if model is not None else os.environ.get(BYOK_MODEL_ENV, "")
        missing = [
            name
            for name, val in (
                (BYOK_BASE_URL_ENV, url),
                (BYOK_API_KEY_ENV, key),
                (BYOK_MODEL_ENV, mdl),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                "BYOK backend is not configured; set "
                + ", ".join(missing)
                + " (fx-1 never hardcodes credentials)"
            )
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise RuntimeError(f"{BYOK_BASE_URL_ENV} must be an http(s) URL, got {url!r}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        self._url = _chat_completions_url(url)
        self._api_key = key
        self._model = mdl
        self._timeout_s = timeout_s

    def complete(self, messages: list[dict[str, str]]) -> str:
        # temperature pinned to 0 — eval/teacher runs must be deterministic;
        # unpinned sampling makes eval results unreproducible across replays.
        body = json.dumps({"model": self._model, "messages": messages, "temperature": 0.0}).encode()
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310 — user-declared BYOK endpoint  # nosec B310
                payload = json.loads(response.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"BYOK endpoint {self._url} unreachable: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "malformed BYOK completion payload: missing choices[0].message.content"
            ) from exc
        if not isinstance(content, str):
            raise RuntimeError(
                f"malformed BYOK completion payload: content is {type(content).__name__}, not str"
            )
        return content


class LocalFx1Backend:
    """A local fx-1 checkpoint. Fail-closed without a valid model card.

    Attestation enforcement: when ``require_signature=True`` (the default once
    FX1_SIGNING_KEY is set), unsigned or signature-mismatched checkpoints
    refuse to serve — the served model must be the model whose card passed
    the gates, verifiably.
    """

    def __init__(
        self, checkpoint_dir: str | Path, *, require_signature: bool | None = None
    ) -> None:
        root = Path(checkpoint_dir)
        card_path = root / "modelcard.json"
        if not card_path.exists():
            raise FileNotFoundError(
                f"no model card at {card_path}; an fx-1 checkpoint without a card is not servable"
            )
        self.card = ModelCard.load(card_path)
        if not self.card.eval_delta.ship_eligible:
            raise RuntimeError(
                f"{self.card.version} failed the ship gate "
                "(honesty/domain/general deltas); refusing to serve"
            )
        if require_signature is None:
            require_signature = bool(os.environ.get("FX1_SIGNING_KEY"))
        if require_signature:
            from fx1.serve.signing import verify_release

            if not verify_release(root):
                raise RuntimeError(
                    "checkpoint release signature missing or invalid; "
                    "fx-1 serves only signed releases when FX1_SIGNING_KEY "
                    "is configured"
                )
        self._root = root

    def complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError(
            "local inference requires the serving stack (vLLM/SGLang with K3 "
            "support); wire it here when the first distilled student ships"
        )


def get_backend(kind: str, **kwargs: object) -> InferenceBackend:
    """Backend factory: ``hosted_k3``, ``local_fx1`` or ``byok``. Fail-closed."""
    backends = {
        "hosted_k3": HostedK3Backend,
        "local_fx1": LocalFx1Backend,
        "byok": OpenAICompatBackend,
    }
    if kind not in backends:
        raise KeyError(f"unknown backend {kind!r}; choose from {sorted(backends)}")
    return cast(InferenceBackend, backends[kind](**kwargs))
