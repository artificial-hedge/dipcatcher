"""Coverage for ``utils/seeds.py`` — process seeding and the optional torch path.

``set_global_seed`` must stay import-safe when torch is absent (the base
install does not pull it), which is the branch the default test environment
would otherwise never exercise.
"""

from __future__ import annotations

import os
import random
import sys
import types
from collections.abc import Iterator

import numpy as np
import pytest

from quant_fund.utils.seeds import set_global_seed


@pytest.fixture(autouse=True)
def _restore_process_rngs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Seeding is process-global; put both RNGs back so no later test inherits it."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available
        self.seeded_all: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def manual_seed_all(self, seed: int) -> None:
        self.seeded_all.append(seed)


def _fake_torch(available: bool) -> tuple[types.ModuleType, list[int], _FakeCuda]:
    module = types.ModuleType("torch")
    seeded: list[int] = []
    cuda = _FakeCuda(available)
    module.manual_seed = seeded.append  # type: ignore[attr-defined]
    module.cuda = cuda  # type: ignore[attr-defined]
    return module, seeded, cuda


def test_set_global_seed_makes_python_and_numpy_draws_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    set_global_seed(1234)
    python_first = [random.random() for _ in range(3)]
    numpy_first = np.random.random(3).tolist()

    set_global_seed(1234)
    assert [random.random() for _ in range(3)] == python_first
    assert np.random.random(3).tolist() == numpy_first


def test_set_global_seed_exports_pythonhashseed_for_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    set_global_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_set_global_seed_is_a_noop_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``None`` entry in sys.modules makes ``import torch`` raise ImportError,
    # which is what a base install without the nn extra looks like.
    monkeypatch.setitem(sys.modules, "torch", None)
    set_global_seed(11)
    assert os.environ["PYTHONHASHSEED"] == "11"


def test_set_global_seed_seeds_cpu_rng_only_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    module, seeded, cuda = _fake_torch(available=False)
    monkeypatch.setitem(sys.modules, "torch", module)
    set_global_seed(21)
    assert seeded == [21]
    assert cuda.seeded_all == []


def test_set_global_seed_seeds_all_cuda_devices_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, seeded, cuda = _fake_torch(available=True)
    monkeypatch.setitem(sys.modules, "torch", module)
    set_global_seed(33)
    assert seeded == [33]
    assert cuda.seeded_all == [33]
