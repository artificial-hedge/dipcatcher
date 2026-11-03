"""fx-1 — the quant LLM fine-tuned from Kimi K3 open weights.

fx-1 (always lowercase) *is the model*: a Kimi K3 fine-tune trained
exclusively on gate-passed, receipt-bound dipcatcher research behavior.
This package is the fx-1 project: corpus construction (``fx1.data``),
evaluation (``fx1.eval``), training (``fx1.train``), serving (``fx1.serve``),
the flagship bench (``fx1.bench``), the reward model (``fx1.reward``), and
the harness bridge (``fx1.harness``) through which dipcatcher builds,
evaluates, and verifies the model. See ``docs/FX1.md``.

``fx1.forecast`` is a separate inference/evaluation harness for an external
price/return model that is not implemented in this package. See
``docs/fx1_harness.md``.
"""

__version__ = "0.4.0"

BASE_MODEL = "moonshotai/Kimi-K3"


def _install_secret_redaction() -> None:
    """Install log redaction without importing the ``quant_fund`` package.

    ``import fx1.serve`` must leave ``quant_fund`` out of ``sys.modules``.
    The harness implementation is stdlib-only, so loading that file by path
    installs the same hook the ``fx1.secret_guard`` re-export calls later.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "quant_fund" / "utils" / "secret_guard.py"
    spec = importlib.util.spec_from_file_location("_fx1_secret_redaction", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load secret redaction from {path}")
    from typing import Any, cast

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = cast(Any, module)
    loaded.install_secret_redaction()


_install_secret_redaction()
