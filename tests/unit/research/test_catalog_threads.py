"""Torch intra-op pin used before the research catalog runs."""

from __future__ import annotations

import pytest


def test_pin_catalog_threads_sets_one_intra_op_thread() -> None:
    torch = pytest.importorskip("torch")
    from quant_fund.utils.seeds import pin_catalog_threads, set_global_seed

    torch.set_num_threads(2)
    pin_catalog_threads()
    assert torch.get_num_threads() == 1
    torch.set_num_threads(2)
    set_global_seed(0)
    assert torch.get_num_threads() == 1
