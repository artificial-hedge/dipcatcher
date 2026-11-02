"""Torch intra-op pin used before the research catalog runs."""

from __future__ import annotations

import pytest


def test_pin_catalog_threads_sets_one_intra_op_thread() -> None:
    torch = pytest.importorskip("torch")
    from quant_fund.utils.seeds import pin_catalog_threads

    pin_catalog_threads()
    assert torch.get_num_threads() == 1
