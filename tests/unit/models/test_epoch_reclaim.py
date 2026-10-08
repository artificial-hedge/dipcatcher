"""Epoch-reclamation honesty tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models import epoch_reclaim as er
from quant_fund.models.epoch_reclaim import bench_epoch_reclaim


def test_trace_flags_eager_free(monkeypatch):
    """An advance that frees objects retired <2 epochs ago must be caught."""

    class EagerEBR(er.EBR):
        def try_advance(self) -> int:
            # defect injection: free every bag regardless of residency
            self.epoch += 1
            doomed = set().union(*self.bags)
            self.freed |= doomed
            return len(doomed)

    monkeypatch.setattr(er, "EBR", EagerEBR)
    assert er._ebr_trace(np.random.RandomState(1)) is False


def test_real_ebr_trace_passes():
    assert er._ebr_trace(np.random.RandomState(7)) is True


def test_bench():
    assert bench_epoch_reclaim()["synthetic_ebr_safe"] == 1.0
