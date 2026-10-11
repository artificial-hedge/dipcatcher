"""Global seed management. GPU determinism is not claimed."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed the process RNGs.

    PYTHONHASHSEED is read by the interpreter at startup, so setting it here
    only affects subprocesses spawned after this call — this interpreter's
    hash order is already fixed.
    """
    random.seed(seed)
    # ``numpy`` and ``torch`` are imported lazily so this module — and every
    # consumer of ``quant_fund.utils`` — does not pay a ~1.5s cold-start for
    # the common case where seeding happens only inside long-running research
    # or training tasks, never during CLI bootstrap.
    import numpy as np

    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
