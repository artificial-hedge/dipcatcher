from __future__ import annotations

import pytest

from fx1.serve.uploads import UploadStore


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_entries": 0, "max_bytes": 1}, "max_entries must be positive"),
        ({"max_entries": -1, "max_bytes": 1}, "max_entries must be positive"),
        ({"max_entries": 1, "max_bytes": 0}, "max_bytes must be positive"),
        ({"max_entries": 1, "max_bytes": -1}, "max_bytes must be positive"),
    ],
)
def test_upload_store_rejects_non_positive_capacity(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        UploadStore(**kwargs)
