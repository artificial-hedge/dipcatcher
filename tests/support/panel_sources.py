"""Materialize the panel sources that dataset identity requires.

``quant_fund.pipeline.artifact_manifest`` binds a fail-closed dataset identity
over the *bytes* of every materialized panel input under ``cfg.data.root``:

    gold/features.parquet, gold/labels.parquet, silver/universe.parquet

That check landed in ``da5c3e78c`` (2026-10-07). Test suites that point
``cfg.data.root`` at an empty ``tmp_path`` — the common pattern for pipeline
tests — therefore began failing with::

    DatasetIdentityError: dataset identity requires materialized panel sources

This is a *test-harness* gap, not a weakening of the check: the requirement is
correct and deliberately fail-closed (it is what makes a later source-data swap
detectable even when panel columns match). The suites simply never got the
matching fixture. Call :func:`materialize_panel_sources` at the top of a
suite's config builder so the identity has real bytes to bind.

The stub frames carry a single column because the identity hashes file bytes;
column content is irrelevant to it. Keep them minimal — this is scaffolding for
the digest, not a fixture that pretends to be a market panel.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["materialize_panel_sources"]


def materialize_panel_sources(data_root: Path) -> Path:
    """Write minimal parquet stubs for every required panel source.

    Returns ``data_root`` so callers can chain. Idempotent — safe to call from
    a fixture that runs more than once for the same directory.
    """
    # Imported lazily: this module is test-only support and must not be part of
    # any production import path.
    import pandas as pd

    from quant_fund.pipeline.artifact_manifest import SOURCE_PANEL_PARTS

    root = Path(data_root)
    for relative in SOURCE_PANEL_PARTS:
        part = root / relative
        part.parent.mkdir(parents=True, exist_ok=True)
        if not part.is_file():
            pd.DataFrame({"_identity_stub": [0.0]}).to_parquet(part)
    return root
