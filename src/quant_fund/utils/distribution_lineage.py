"""Foundation-layer leaf for the crown-jewel distribution digest.

``fx1.doctor`` must not import ``quant_fund.research`` (harness-surface
allowlist). This wrapper lazy-imports the library check so the doctor
reports the same ``jewels_intact`` / digest as
``quant_fund.research.crown_jewels.distribution_lineage_digest``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def distribution_lineage_digest(root: Path | str = ".") -> dict[str, Any]:
    """Same payload as the crown-jewels library digest (lazy research import)."""
    from quant_fund.research.crown_jewels import (
        distribution_lineage_digest as _library_digest,
    )

    return _library_digest(root)
