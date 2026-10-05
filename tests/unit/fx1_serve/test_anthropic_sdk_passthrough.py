"""The SDK audit must notice if the translator drops claimed parameters."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import api
from fx1.serve.anthropic_sdk_audit import _probe_messages, _sdk_client


def test_parameter_probes_detect_dropped_translation_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("FX1_API_STATE_DIR", raising=False)
    monkeypatch.setenv("FX1_API_RECEIPTS_DIR", str(tmp_path))
    translate = api.anthropic_to_openai

    def drop_claimed_parameters(body: Any) -> dict[str, Any]:
        translated: dict[str, Any] = translate(body)
        for key in ("user", "stop", "temperature", "top_p"):
            translated.pop(key, None)
        return translated

    monkeypatch.setattr(api, "anthropic_to_openai", drop_claimed_parameters)

    async def check() -> None:
        with _sdk_client() as (sdk, stub, _http):
            try:
                results: dict[str, Any] = {}
                await _probe_messages(sdk, stub, results)
            finally:
                await sdk.close()
        for probe in ("metadata", "stop_sequences", "temperature", "top_p"):
            assert results[f"sdk_msg_{probe}"] is False, f"{probe} silently dropped"
        assert results["sdk_msg_create"] is True

    asyncio.run(check())
