"""CI-budget optional blobs are reused across processes and skipped in production."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from quant_fund.research import agent
from quant_fund.research.ci_blob_cache import blob_dir, source_stamp


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DIP_SESSION_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("DIP_DISABLE_SESSION_CACHE", raising=False)
    agent._CI_OPTIONAL_BLOBS.clear()
    yield tmp_path
    agent._CI_OPTIONAL_BLOBS.clear()


def test_ci_budget_reuses_blobs_from_disk(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "ci")
    calls = {"n": 0}

    def fake() -> dict[str, dict[str, float]]:
        calls["n"] += 1
        return {"sentinel": {"x": 1.0}}

    monkeypatch.setattr(agent, "_build_live_optional_family_blobs", fake)
    first = agent._live_optional_family_blobs()
    assert first == {"sentinel": {"x": 1.0}}
    assert calls["n"] == 1
    agent._CI_OPTIONAL_BLOBS.clear()
    second = agent._live_optional_family_blobs()
    assert second == {"sentinel": {"x": 1.0}}
    assert calls["n"] == 1
    stored = list((isolated_cache / "ci-optional" / source_stamp()).glob("ci.pkl"))
    assert stored and stored[0].is_file()


def test_full_budget_does_not_touch_the_disk_cache(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "full")
    calls = {"n": 0}

    def fake() -> dict[str, dict[str, float]]:
        calls["n"] += 1
        return {"sentinel": {"x": 2.0}}

    monkeypatch.setattr(agent, "_build_live_optional_family_blobs", fake)
    assert agent._live_optional_family_blobs()["sentinel"]["x"] == 2.0
    assert agent._live_optional_family_blobs()["sentinel"]["x"] == 2.0
    assert calls["n"] == 2
    assert not (isolated_cache / "ci-optional").exists()


def test_disk_cache_serves_a_cold_process(
    isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_SCORECARD_BUDGET", "ci")
    monkeypatch.setattr(
        agent,
        "_build_live_optional_family_blobs",
        lambda: {"sentinel": {"x": 3.0}},
    )
    assert agent._live_optional_family_blobs()["sentinel"]["x"] == 3.0
    script = textwrap.dedent(
        f"""
        import os
        os.environ["DIP_SCORECARD_BUDGET"] = "ci"
        os.environ["DIP_SESSION_CACHE_DIR"] = {str(isolated_cache)!r}
        os.environ["DIP_DISABLE_SESSION_CACHE"] = ""
        import quant_fund.research.agent as agent

        def boom():
            raise RuntimeError("cold process rebuilt the CI blobs")

        agent._build_live_optional_family_blobs = boom
        blobs = agent._live_optional_family_blobs()
        assert blobs == {{"sentinel": {{"x": 3.0}}}}
        print("HIT")
        """
    )
    env = os.environ.copy()
    env["DIP_SCORECARD_BUDGET"] = "ci"
    env["DIP_SESSION_CACHE_DIR"] = str(isolated_cache)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert proc.returncode == 0, proc.stderr
    assert "HIT" in proc.stdout
    assert blob_dir().name == source_stamp()
