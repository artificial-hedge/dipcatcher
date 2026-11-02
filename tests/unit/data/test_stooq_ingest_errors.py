"""Malformed-response error paths at the Stooq ingestion boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.data.adapters import stooq
from quant_fund.data.sources.adapters import StooqSource
from quant_fund.data.sources.base import SourceError

_MALFORMED_RESPONSE = "<html><body>not a Stooq CSV</body></html>"


def test_download_malformed_response_is_empty_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(stooq, "fetch_stooq_csv", lambda symbol, **kwargs: _MALFORMED_RESPONSE)
    root = tmp_path / "stooq-lake"

    result = stooq.download_stooq_universe(
        root,
        names=(("TEST", "test.us"),),
        pause_s=0.0,
        max_workers=1,
    )

    assert result["status"] == "empty"
    assert result["n_names"] == 0
    assert result["errors"] == {}
    assert not root.exists()


def test_source_malformed_response_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stooq, "fetch_stooq_csv", lambda symbol, **kwargs: _MALFORMED_RESPONSE)

    with pytest.raises(SourceError, match="returned no bars"):
        StooqSource().fetch(
            names="TEST:test.us",
            strict=False,
            pause_s=0.0,
            max_workers=1,
        )
