"""``fx1 harness`` subcommands — wire surface for the dipcatcher harness.

Command implementations live in ``cli_harness_*.py`` and register on the
shared ``harness_app`` at import time.
"""

from __future__ import annotations

from fx1.cli_common import harness_app

# ruff: noqa: E402, F401, I001
import fx1.cli_harness_ops as _cli_harness_ops
import fx1.cli_harness_jobs as _cli_harness_jobs
import fx1.cli_harness_ft as _cli_harness_ft
import fx1.cli_harness_chat as _cli_harness_chat
import fx1.cli_harness_vs as _cli_harness_vs

__all__ = ["harness_app"]
