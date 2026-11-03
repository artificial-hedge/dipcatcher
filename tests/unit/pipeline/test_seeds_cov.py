"""Behavioral coverage for ``quant_fund.utils.seeds.set_global_seed``.

The module seeds the stdlib RNG, NumPy's legacy global RandomState, and
``PYTHONHASHSEED`` unconditionally, then seeds torch only when it is
importable (torch is an optional accelerator — GPU determinism is not
claimed).
"""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterator

import numpy as np
import pytest

from quant_fund.utils.seeds import set_global_seed


@pytest.fixture(autouse=True)
def _restore_process_rng() -> Iterator[None]:
    """set_global_seed mutates process RNGs; later tests must not see that."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    hashseed = os.environ.get("PYTHONHASHSEED")
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch is not None else None
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    if hashseed is None:
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        os.environ["PYTHONHASHSEED"] = hashseed
    if torch is not None and torch_state is not None:
        torch.set_rng_state(torch_state)


def test_seeds_stdlib_numpy_and_pythonhashseed() -> None:
    set_global_seed(42)

    assert os.environ["PYTHONHASHSEED"] == "42"
    assert random.random() == random.Random(42).random()
    assert np.random.rand() == np.random.RandomState(42).rand()


def test_repeated_seeding_reproduces_sequences() -> None:
    set_global_seed(7)
    first = (random.random(), np.random.rand())
    set_global_seed(7)
    assert first == (random.random(), np.random.rand())


def test_seeds_torch_rng_when_torch_is_installed() -> None:
    torch = pytest.importorskip("torch")

    set_global_seed(123)
    expected = torch.rand(4)
    set_global_seed(123)

    assert torch.equal(expected, torch.rand(4))


def test_seeds_all_cuda_devices_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[str] = []
    # torch.manual_seed itself delegates to torch.cuda.manual_seed_all, so
    # stub it to observe only the explicit CUDA-guarded call in seeds.py.
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(f"cpu:{seed}"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(f"cuda:{seed}"))

    set_global_seed(77)

    assert calls == ["cpu:77", "cuda:77"]


def test_skips_cuda_seeding_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[str] = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(f"cpu:{seed}"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(f"cuda:{seed}"))

    set_global_seed(77)

    assert calls == ["cpu:77"]


def test_still_seeds_stdlib_and_numpy_when_torch_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None entry in sys.modules makes `import torch` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)

    set_global_seed(9)

    assert random.random() == random.Random(9).random()
    assert np.random.rand() == np.random.RandomState(9).rand()
