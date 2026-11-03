"""Global seed management. GPU determinism is not claimed."""

from __future__ import annotations

import os
import random

import numpy as np


def pin_catalog_threads() -> None:
    """One intra-op torch thread. A checkout without torch is a no-op.

    PyTorch keeps its own pool at the host CPU count after the first parallel
    region. Research calls this from ``set_global_seed`` before any catalog
    bench, so one ``run_research`` does not oversubscribe every small op.
    """
    try:
        import torch
    except ImportError:
        return
    setter = getattr(torch, "set_num_threads", None)
    if callable(setter):
        setter(1)


def set_global_seed(seed: int) -> None:
    """Seed the process RNGs.

    PYTHONHASHSEED is read by the interpreter at startup, so setting it here
    only affects subprocesses spawned after this call — this interpreter's
    hash order is already fixed.
    """
    pin_catalog_threads()
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
