"""SYNTHETIC cache-hit regression: unchanged receipt inventories are not resorted."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fx1.serve.receipt_store import ReceiptIndex


def test_unchanged_store_does_not_resort_the_inventory(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"receipt_sha256": "a" * 64}), encoding="utf-8")
    index = ReceiptIndex(tmp_path)
    assert index.items() == [("a" * 64, path)]
    with patch(
        "fx1.serve.receipt_store.sorted",
        side_effect=AssertionError("unchanged inventory sorted again"),
        create=True,
    ):
        assert index.items() == [("a" * 64, path)]
