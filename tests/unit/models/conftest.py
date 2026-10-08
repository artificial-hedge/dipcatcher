"""Deterministic compute ambient for the models suite.

Several product modules pin ``torch.set_num_threads(1)`` for
reproducibility and never restore it, so the ambient thread count at a
bench's entry depends on suite order — and floating-point reduction
order follows thread count. Pin a canonical single-threaded ambient for
every test in this directory so seeded benches are deterministic
regardless of what ran before.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _canonical_torch_threads() -> None:
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
