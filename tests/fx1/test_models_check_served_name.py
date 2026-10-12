"""A served-model id from the server is untrusted text and stays on one clean line."""

from __future__ import annotations

from fx1.interactive.models_check import _public_name


def test_newline_and_escape_in_a_served_id_collapse_to_one_line() -> None:
    hostile = "fx1\nFORGED ROW  verified\x1b]0;owned\x07\x1b[2J tail"
    shown = _public_name(hostile)
    assert shown is not None
    assert "\n" not in shown
    assert "\x1b" not in shown
    assert "owned" not in shown
    assert shown.startswith("fx1")


def test_none_and_empty_stay_none() -> None:
    assert _public_name(None) is None
    assert _public_name("\x1b[2J\n") is None


def test_long_ids_are_capped() -> None:
    shown = _public_name("m" * 500)
    assert shown is not None
    assert len(shown) == 128
