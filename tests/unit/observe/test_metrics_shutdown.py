"""Bounded metrics-server shutdown behavior."""

from __future__ import annotations

import pytest

from quant_fund.observe import serve


class _SurvivingThread:
    def join(self, timeout: float) -> None:
        assert timeout == 2

    def is_alive(self) -> bool:
        return True


def test_stop_metrics_server_fails_if_thread_survives_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serve, "_SERVER", None)
    monkeypatch.setattr(serve, "_THREAD", _SurvivingThread())
    with pytest.raises(RuntimeError, match="did not stop within 2 seconds"):
        serve.stop_metrics_server()
