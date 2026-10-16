"""The canon writer must commit LF bytes on Windows.

``Path.write_text`` otherwise turns ``\\n`` into ``\\r\\n``. ``--check``
reads with universal newlines, so that CRLF file still compares equal and
can land in git.
"""

from __future__ import annotations

from pathlib import Path

import scripts.canon_qualify as canon_qualify


def test_write_utf8_lf_does_not_emit_cr(tmp_path: Path) -> None:
    target = tmp_path / "summary.json"
    canon_qualify._write_utf8_lf(target, '{\n  "ok": true\n}\n')
    raw = target.read_bytes()
    assert b"\r" not in raw
    assert raw == b'{\n  "ok": true\n}\n'
