"""Architecture freshness checks report invalid UTF-8 as stale artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import gen_arch_diagrams as diagrams


@pytest.mark.parametrize(
    "corrupt_relative",
    ["docs/architecture/manifest.json", "docs/ARCHITECTURE_ATLAS.md"],
)
def test_invalid_utf8_is_reported_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    corrupt_relative: str,
) -> None:
    manifest_relative = "docs/architecture/manifest.json"
    artifacts = {manifest_relative: '{"ok": true}\n'}
    monkeypatch.setattr(diagrams, "verify_diagram_bindings", lambda _root: [])
    monkeypatch.setattr(diagrams, "generated_artifacts", lambda _root: artifacts)
    monkeypatch.setattr(
        diagrams, "generated_blocks", lambda _artifacts: {"example": "body"}
    )

    manifest = tmp_path / manifest_relative
    manifest.parent.mkdir(parents=True)
    manifest.write_text(artifacts[manifest_relative], encoding="utf-8")
    atlas = tmp_path / diagrams.ATLAS_DOC
    atlas.write_text(
        "<!-- BEGIN GENERATED: example -->\n"
        "body\n"
        "<!-- END GENERATED: example -->",
        encoding="utf-8",
    )
    (tmp_path / corrupt_relative).write_bytes(b"invalid UTF-8: " + bytes([0xFF]))

    assert diagrams.stale_artifacts(tmp_path) == [corrupt_relative]
    assert diagrams.main(["--check", "--root", str(tmp_path)]) == 1
    assert f"  {corrupt_relative}\n" in capsys.readouterr().out
