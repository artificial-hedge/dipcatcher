"""Foundation-layer leaf for the kernel-lineage checkout root.

``fx1.interactive.superpower`` must not import ``quant_fund.audit``
(harness-surface allowlist). This wrapper lazy-imports the library
helper so receipt roots still come from
``quant_fund.audit.kernel_lineage.default_repo_root``.
"""

from __future__ import annotations

from pathlib import Path


def default_repo_root() -> Path:
    """Same checkout root as the kernel-lineage library helper."""
    from quant_fund.audit.kernel_lineage import default_repo_root as _library_root

    return _library_root()
