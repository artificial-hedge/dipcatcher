"""SYNTHETIC lifecycle regression checks for absolute response replay cursors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fx1.serve import api


@pytest.mark.parametrize("initial_status", ["queued", "in_progress"])
@pytest.mark.parametrize("terminal_status", ["cancelled", "failed"])
@pytest.mark.parametrize("skip", [0, 1, 2, 3, 4, 1001])
def test_response_replay_preserves_absolute_cursor_during_follow(
    monkeypatch: pytest.MonkeyPatch,
    initial_status: str,
    terminal_status: str,
    skip: int,
) -> None:
    envelope = {
        "id": "resp_cursor_regression",
        "object": "response",
        "created_at": 1,
        "background": True,
        "output": [],
        "usage": None,
    }

    class LifecycleStore:
        reads = 0

        def get(self, response_id: str) -> dict[str, Any]:
            assert response_id == envelope["id"]
            self.reads += 1
            status = initial_status if self.reads == 1 else terminal_status
            return {**envelope, "status": status}

    store: Any = LifecycleStore()
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _: None))
    frames = list(
        api._responses_replay_frames(
            store,
            str(envelope["id"]),
            skip=skip,
            timeout_s=1.0,
            keepalive_s=0.0,
        )
    )
    ids = [
        int(line.removeprefix("id: "))
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("id: ")
    ]
    # A terminal background response without output has the four absolute
    # event IDs: created=0, queued=1, in_progress=2, terminal=3. A cursor
    # beyond the currently available prelude must never move backwards.
    assert ids == list(range(skip, 4))
    assert store.reads == 2
