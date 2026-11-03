"""The CI blob tarball round-trips only when the stamp still matches."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from quant_fund.research.ci_blob_cache import load_blobs, source_stamp, store_blobs

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci_optional_blobs.py"


def _run(cache: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DIP_SESSION_CACHE_DIR"] = str(cache)
    env.pop("DIP_DISABLE_SESSION_CACHE", None)
    env["DIP_SCORECARD_BUDGET"] = "ci"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


@pytest.fixture
def filled_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "src"
    monkeypatch.setenv("DIP_SESSION_CACHE_DIR", str(cache))
    monkeypatch.delenv("DIP_DISABLE_SESSION_CACHE", raising=False)
    assert store_blobs("ci", {"sentinel": {"x": 1.0}})
    return cache


def test_pack_install_roundtrip(filled_cache: Path, tmp_path: Path) -> None:
    archive = tmp_path / "ci-optional.tar.gz"
    packed = _run(filled_cache, "pack", "--out", str(archive))
    assert packed.returncode == 0, packed.stderr
    assert source_stamp() in packed.stdout
    dest = tmp_path / "dst"
    installed = _run(dest, "install", "--archive", str(archive))
    assert installed.returncode == 0, installed.stderr
    os.environ["DIP_SESSION_CACHE_DIR"] = str(dest)
    try:
        loaded = load_blobs("ci")
    finally:
        os.environ["DIP_SESSION_CACHE_DIR"] = str(filled_cache)
    assert loaded == {"sentinel": {"x": 1.0}}


def test_install_refuses_a_foreign_stamp(filled_cache: Path, tmp_path: Path) -> None:
    archive = tmp_path / "ci-optional.tar.gz"
    assert _run(filled_cache, "pack", "--out", str(archive)).returncode == 0
    with tarfile.open(archive, "r:gz") as src:
        member = src.getmembers()[0]
        payload = src.extractfile(member)
        assert payload is not None
        data = payload.read()
    foreign = "a" * len(source_stamp())
    assert foreign != source_stamp()
    foreign_archive = tmp_path / "foreign.tar.gz"
    info = tarfile.TarInfo(name=f"ci-optional/{foreign}/ci.pkl")
    info.size = len(data)
    with tarfile.open(foreign_archive, "w:gz") as dst:
        dst.addfile(info, io.BytesIO(data))
    installed = _run(tmp_path / "dst", "install", "--archive", str(foreign_archive))
    assert installed.returncode == 2
    assert "does not match" in installed.stderr
    assert not (tmp_path / "dst" / "ci-optional" / source_stamp() / "ci.pkl").exists()


def test_stamp_matches_the_loader(tmp_path: Path) -> None:
    proc = _run(tmp_path, "stamp")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == source_stamp()


def test_pack_without_a_pickle_fails(tmp_path: Path) -> None:
    proc = _run(tmp_path / "empty", "pack", "--out", str(tmp_path / "out.tar.gz"))
    assert proc.returncode == 2
    assert "missing" in proc.stderr
