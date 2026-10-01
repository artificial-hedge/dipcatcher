"""PIT vault immutability: asof(t) is a function of t alone — never of when
you ask. Appends and restatements land at later ``known_at`` revisions, so a
query at an old watermark must return byte-identical content forever. The one
legitimate mutation is a *backdated* correction, gated behind the opt-in
``monotonic_known_at`` dataset flag (warn-only by default).
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from quant_fund.pit import PitVault, VaultError
from tests.unit.pit.test_vault_enforcement import T0, _frame


def _rows(frame) -> list[tuple]:
    return frame.sort(frame.columns).rows()


@pytest.fixture()
def vault(tmp_path) -> PitVault:
    v = PitVault(tmp_path / "pit")
    v.create_dataset("silver/bars")
    return v


def test_asof_immutable_under_later_appends(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0), ("A", 1, 1, 2.0)]))
    before = _rows(vault.asof("silver/bars", T0 + timedelta(days=1)).frame)
    vault.append("silver/bars", _frame([("A", 2, 5, 3.0), ("B", 0, 5, 9.0)]))
    assert _rows(vault.asof("silver/bars", T0 + timedelta(days=1)).frame) == before


def test_asof_immutable_under_restatement(vault: PitVault) -> None:
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0)]))
    before = _rows(vault.asof("silver/bars", T0 + timedelta(days=1)).frame)
    vault.restate(
        "silver/bars",
        _frame([("A", 0, 7, 1.5)]),
        known_at=T0 + timedelta(days=7),
    )
    # The old watermark still sees the superseded row — history was appended,
    # not rewritten.
    assert _rows(vault.asof("silver/bars", T0 + timedelta(days=1)).frame) == before
    # …and the post-correction watermark sees the corrected value.
    corrected = vault.asof("silver/bars", T0 + timedelta(days=8)).frame
    assert corrected.get_column("close").to_list() == [1.5]


def test_visibility_monotone_in_watermark(vault: PitVault) -> None:
    """Rows observable at t1 remain observable at every t2 > t1 — the view
    only grows."""
    vault.append("silver/bars", _frame([("A", 0, 0, 1.0), ("A", 1, 1, 2.0)]))
    vault.append("silver/bars", _frame([("A", 2, 5, 3.0), ("B", 1, 6, 4.0)]))
    early = vault.asof("silver/bars", T0 + timedelta(days=1)).frame
    late = vault.asof("silver/bars", T0 + timedelta(days=10)).frame
    early_keys = set(early.select(["security_id", "event_time"]).iter_rows())
    late_keys = set(late.select(["security_id", "event_time"]).iter_rows())
    assert early_keys <= late_keys
    # Earlier event rows still carry their earliest-known values.
    early_map = {
        (r[0], r[1]): r[-1]
        for r in early.select(["security_id", "event_time", "close"]).iter_rows()
    }
    for r in late.select(["security_id", "event_time", "close"]).iter_rows():
        if (r[0], r[1]) in early_map:
            assert r[2] == early_map[(r[0], r[1])]


def test_backdated_correction_refused_under_monotonic_flag(tmp_path) -> None:
    """monotonic_known_at: a restate cannot launder an old watermark — an
    append whose min known_at precedes the dataset max is refused outright."""
    v = PitVault(tmp_path / "pit")
    v.create_dataset("silver/bars", monotonic_known_at=True)
    v.append("silver/bars", _frame([("A", 0, 5, 1.0)]))
    before = _rows(v.asof("silver/bars", T0 + timedelta(days=6)).frame)
    with pytest.raises(VaultError):
        v.restate(
            "silver/bars",
            _frame([("A", 0, 2, 9.9)]),
            known_at=T0 + timedelta(days=2),
        )
    assert _rows(v.asof("silver/bars", T0 + timedelta(days=6)).frame) == before


def test_backdated_correction_before_original_never_supersedes(
    vault: PitVault,
) -> None:
    """Under LATEST_KNOWN a correction stamped *before* the original's
    known_at can never win — the later-known original stays visible."""
    vault.append("silver/bars", _frame([("A", 0, 5, 1.0)]))
    vault.restate(
        "silver/bars",
        _frame([("A", 0, 2, 9.9)]),
        known_at=T0 + timedelta(days=2),
    )
    view = vault.asof("silver/bars", T0 + timedelta(days=6)).frame
    assert view.get_column("close").to_list() == [1.0]


def test_midrange_backdated_correction_rewrites_watermark_warn_only(
    vault: PitVault,
) -> None:
    """The real laundering class: stamp known_at *between* revisions and
    asof() in that window retroactively changes — warn-only by default."""
    vault.append("silver/bars", _frame([("A", 0, 1, 1.0)]))
    vault.append("silver/bars", _frame([("B", 0, 5, 2.0)]))
    vault.restate(
        "silver/bars",
        _frame([("A", 0, 3, 9.9)]),
        known_at=T0 + timedelta(days=3),
    )
    view = vault.asof("silver/bars", T0 + timedelta(days=4)).frame
    a_row = view.filter(pl.col("security_id") == "A")
    assert a_row.get_column("close").to_list() == [9.9]


def test_midrange_backdated_correction_refused_under_monotonic_flag(
    tmp_path,
) -> None:
    """monotonic_known_at converts the laundering class into a hard refusal —
    no retroactive rewrite of any historical watermark."""
    v = PitVault(tmp_path / "pit")
    v.create_dataset("silver/bars", monotonic_known_at=True)
    v.append("silver/bars", _frame([("A", 0, 1, 1.0)]))
    v.append("silver/bars", _frame([("B", 0, 5, 2.0)]))
    before = _rows(v.asof("silver/bars", T0 + timedelta(days=4)).frame)
    with pytest.raises(VaultError):
        v.restate(
            "silver/bars",
            _frame([("A", 0, 3, 9.9)]),
            known_at=T0 + timedelta(days=3),
        )
    assert _rows(v.asof("silver/bars", T0 + timedelta(days=4)).frame) == before
