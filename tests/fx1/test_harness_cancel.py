"""Owned harness children stop when the session cancels them."""

from __future__ import annotations

import sys
import threading

from fx1.harness import popen_cancellable


def test_cancel_stops_owned_child() -> None:
    cancel = threading.Event()

    def trip() -> None:
        cancel.wait(0.05)
        cancel.set()

    threading.Thread(target=trip, daemon=True).start()
    code, _out, err = popen_cancellable(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=10,
        cancel_event=cancel,
    )
    assert code == 130
    assert "cancelled" in err


def test_finished_child_returns_output() -> None:
    code, out, err = popen_cancellable(
        [sys.executable, "-c", "print('ok')"],
        timeout_s=10,
        cancel_event=None,
    )
    assert code == 0
    assert "ok" in out
    assert err == ""
