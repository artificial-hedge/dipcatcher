"""Environment validation for local inference backends."""

from __future__ import annotations

import pytest

from fx1.serve.backends import LOCAL_TIMEOUT_S_ENV, _env_float


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
def test_nonfinite_timeout_env_fails_closed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(LOCAL_TIMEOUT_S_ENV, value)
    with pytest.raises(ValueError, match=f"{LOCAL_TIMEOUT_S_ENV} must be finite and positive"):
        _env_float(LOCAL_TIMEOUT_S_ENV, None, 120.0)
