from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest

from quant_fund.live.brokers import AlpacaBroker


def _row(index: int) -> dict[str, str]:
    when = datetime(2024, 1, 3, 20, 0, tzinfo=UTC) + timedelta(seconds=index)
    return {
        "id": f"fill-{index:04d}",
        "order_id": "ord-1",
        "side": "buy",
        "qty": "1",
        "price": "100",
        "symbol": "AAPL",
        "transaction_time": when.isoformat(),
    }


class _Pages:
    def __init__(self, rows: list[dict[str, str]], *, repeat_first: bool = False) -> None:
        self.rows = rows
        self.repeat_first = repeat_first
        self.tokens: list[str | None] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        assert "/v2/account/activities/FILL?" in url
        token = parse_qs(urlsplit(url).query).get("page_token", [None])[0]
        self.tokens.append(token)
        if self.repeat_first:
            chunk = self.rows[:100]
        else:
            start = 0
            if token is not None:
                start = next(i for i, row in enumerate(self.rows) if row["id"] == token) + 1
            chunk = self.rows[start : start + 100]
        return 200, json.dumps(chunk).encode()


def _broker(pages: _Pages) -> AlpacaBroker:
    return AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=pages)


@pytest.mark.parametrize(("total", "expected_calls"), [(100, 2), (150, 2), (250, 3)])
def test_fill_reconciliation_reads_every_page(total: int, expected_calls: int) -> None:
    rows = [_row(index) for index in range(total, 0, -1)]
    pages = _Pages(rows)

    result = _broker(pages).fills(broker_to_client={"ord-1": "dc-1"})

    assert [fill.fill_id for fill in result.fills] == [
        f"fill-{index:04d}" for index in range(1, total + 1)
    ]
    assert result.truncated is False
    assert len(pages.tokens) == expected_calls
    assert pages.tokens[0] is None
    assert pages.tokens[1] == rows[99]["id"]


def test_repeated_page_token_stops_and_reports_truncation() -> None:
    pages = _Pages([_row(index) for index in range(100, 0, -1)], repeat_first=True)

    result = _broker(pages).fills(broker_to_client={"ord-1": "dc-1"})

    assert result.truncated is True
    assert pages.tokens == [None, "fill-0001"]


def test_full_page_without_continuation_id_reports_truncation() -> None:
    rows = [_row(index) for index in range(100, 0, -1)]
    rows[-1].pop("id")
    pages = _Pages(rows)

    result = _broker(pages).fills(broker_to_client={"ord-1": "dc-1"})

    assert result.truncated is True
    assert pages.tokens == [None]
    assert len(result.fills) == 99
