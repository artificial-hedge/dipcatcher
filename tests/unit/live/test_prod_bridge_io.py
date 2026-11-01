from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.live.bridge import BridgeError, DropBridge


def test_drop_bridge_normalizes_event_read_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbound = tmp_path / "inbound"
    inbound.mkdir()
    event = inbound / "event.json"
    event.write_text("{}", encoding="utf-8")
    original = Path.read_text

    def fail_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == event:
            raise OSError("simulated I/O failure")
        return original(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(BridgeError, match="event.json: unreadable event"):
        DropBridge(tmp_path).poll()
