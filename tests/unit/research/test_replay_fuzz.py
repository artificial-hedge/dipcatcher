"""Adversarial mutations of replay carriers — every tamper class must fail closed.

The carrier seal is integrity, not authenticity: sha256 resealing is free, so
each mutation below is re-sealed honestly before the replay. The pins inside
``replay`` (argv, artifact digests, input tapes) are what must hold.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from quant_fund.research.replay_proof import run_replay

# Replays write under the replay output dir, the only place replay may clear
# a stale artifact before argv runs.
_ART = "data/metadata/replay/artifact.json"


def _tracked(root: Path, name: str, source: str) -> None:
    """Write a lane script and ``git add`` it: replay runs tracked code only."""
    (root / name).write_text(source)
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "--", name], cwd=root, check=True)


def _emit_script(root: Path) -> None:
    (root / _ART).parent.mkdir(parents=True, exist_ok=True)
    _tracked(
        root,
        "emit.py",
        f"import json, pathlib\npathlib.Path({_ART!r}).write_text(json.dumps({{'fixed': True}}))\n",
    )


def _carrier(root: Path, **replay_overrides: object) -> Path:
    artifact = b'{"fixed": true}'
    replay: dict[str, object] = {
        "argv": ["emit.py"],
        "artifacts": [{"path": _ART, "sha256": hashlib.sha256(artifact).hexdigest()}],
    }
    replay.update(replay_overrides)
    body: dict[str, object] = {
        "schema": "replay_manifest.v1",
        "kind": "replay_manifest",
        "data_label": "SYNTHETIC",
        "replay": replay,
    }
    path = root / "carrier.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def _replay(tmp_path: Path, **overrides: object) -> dict[str, object]:
    _emit_script(tmp_path)
    carrier = _carrier(tmp_path, **overrides)
    return run_replay(carrier, root=tmp_path, timeout_s=60)


def test_sha_mutation_fails(tmp_path: Path) -> None:
    body = _replay(
        tmp_path,
        artifacts=[{"path": _ART, "sha256": "00" * 32}],
    )
    assert body["verdict"] == "fail"
    assert body["artifacts"][0]["match"] is False


def test_artifact_path_escape_fails(tmp_path: Path) -> None:
    body = _replay(
        tmp_path,
        artifacts=[{"path": "../escaped.json", "sha256": "00" * 32}],
    )
    assert body["verdict"] == "fail"
    assert body["artifacts"][0]["note"] == "path_escapes_root"


def test_argv_mutation_leaves_artifact_missing(tmp_path: Path) -> None:
    # argv now runs a no-op script; the pinned artifact must not appear
    _tracked(tmp_path, "noop.py", "pass\n")
    body = _replay(tmp_path, argv=["noop.py"])
    assert body["verdict"] == "fail"
    assert body["artifacts"][0]["note"] == "artifact_missing"


def test_stale_preplaced_artifact_does_not_satisfy(tmp_path: Path) -> None:
    # pre-place the pinned bytes; argv writes nothing -> stale copy must be
    # deleted pre-spawn, so the artifact is missing, not satisfied
    _tracked(tmp_path, "noop.py", "pass\n")
    (tmp_path / _ART).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / _ART).write_bytes(b'{"fixed": true}')
    body = _replay(tmp_path, argv=["noop.py"])
    assert body["verdict"] == "fail"
    assert body["artifacts"][0]["note"] == "artifact_missing"


def test_cwd_escape_fails(tmp_path: Path) -> None:
    body = _replay(tmp_path, cwd="..")
    assert body["verdict"] == "fail"
    assert str(body["spawn_error"]).startswith("cwd_escapes_root")


def test_cwd_missing_fails(tmp_path: Path) -> None:
    body = _replay(tmp_path, cwd="no_such_dir")
    assert body["verdict"] == "fail"
    assert str(body["spawn_error"]).startswith("cwd_missing")


def test_input_tape_bad_sha_skips_execution(tmp_path: Path) -> None:
    tape = tmp_path / "tape.bin"
    tape.write_bytes(b"real tape bytes")
    body = _replay(
        tmp_path,
        input_tapes=[{"path": "tape.bin", "sha256": "11" * 32}],
    )
    assert body["verdict"] == "fail"
    assert body["spawn_error"] == "inputs_not_verified"
    assert body["inputs_ok"] is False
    assert body["exit_code"] is None


def test_input_tape_missing_manifest_skips_execution(tmp_path: Path) -> None:
    body = _replay(
        tmp_path,
        input_tapes=[{"manifest": "data/manifests/nope.json"}],
    )
    assert body["verdict"] == "fail"
    assert body["spawn_error"] == "inputs_not_verified"


def test_committed_artifact_overwrite_blocked(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / _ART
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("old committed bytes")
    subprocess.run(["git", "add", _ART], cwd=tmp_path, check=True)
    body = _replay(tmp_path)
    assert body["verdict"] == "fail"
    assert str(body["spawn_error"]).startswith("artifact_overwrites_committed")
    assert tracked.read_text() == "old committed bytes"


def test_nonzero_exit_fails(tmp_path: Path) -> None:
    _tracked(
        tmp_path,
        "die.py",
        "import json, pathlib, sys\n"
        f"pathlib.Path({_ART!r}).write_text(json.dumps({{'fixed': True}}))\n"
        "sys.exit(3)\n",
    )
    body = _replay(tmp_path, argv=["die.py"])
    assert body["verdict"] == "fail"
    assert body["exit_code"] == 3


def test_replay_lane_tmpdir_is_under_run_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lane children must not inherit the runner-global TMPDIR."""
    foreign_tmp = tmp_path / "foreign_tmp"
    foreign_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(foreign_tmp))
    lane = tmp_path / "lane_cwd"
    lane.mkdir()
    _tracked(
        tmp_path,
        "lane_cwd/emit_tmp.py",
        "import json, os\n"
        "from pathlib import Path\n"
        "p = Path('out')\n"
        "p.mkdir(parents=True, exist_ok=True)\n"
        "(p / 'artifact.json').write_text(json.dumps({'tmpdir': os.environ['TMPDIR']}), encoding='utf-8')\n",
    )
    artifact_path = lane / "out" / "artifact.json"
    body = _replay(
        tmp_path,
        argv=["emit_tmp.py"],
        cwd="lane_cwd",
        artifacts=[{"path": "out/artifact.json", "sha256": "00" * 32}],
    )
    assert body["exit_code"] == 0
    observed = json.loads(artifact_path.read_text())
    assert observed["tmpdir"].endswith("lane_cwd/.replay_lane_tmp")
    assert observed["tmpdir"] != str(foreign_tmp)


def test_inherited_home_cache_does_not_enter_lane_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poisoned parent HOME (CI ``~/.cache/dipcatcher``) must not be
    visible to the lane child. Passing-alone vs with-neighbours: the same
    SHA wrote the pinned name in isolation and a sibling hash after a
    neighbour had populated HOME / a parent ``.git``.
    """
    poison_home = tmp_path / "poison_home"
    cache = poison_home / ".cache" / "dipcatcher"
    cache.mkdir(parents=True)
    (cache / "seed").write_text("LEAK")
    monkeypatch.setenv("HOME", str(poison_home))
    monkeypatch.setenv("USERPROFILE", str(poison_home))
    lane = tmp_path / "lane_cwd"
    lane.mkdir()
    _tracked(
        tmp_path,
        "lane_cwd/emit_home.py",
        "import json, os\n"
        "from pathlib import Path\n"
        "p = Path('out')\n"
        "p.mkdir(parents=True, exist_ok=True)\n"
        "home = Path.home()\n"
        "seed = home / '.cache' / 'dipcatcher' / 'seed'\n"
        "(p / 'artifact.json').write_text(\n"
        "    json.dumps({'home': str(home), 'leak': seed.exists(), "
        "'ceiling': os.environ.get('GIT_CEILING_DIRECTORIES')}),\n"
        "    encoding='utf-8',\n"
        ")\n",
    )
    artifact_path = lane / "out" / "artifact.json"
    body = _replay(
        tmp_path,
        argv=["emit_home.py"],
        cwd="lane_cwd",
        artifacts=[{"path": "out/artifact.json", "sha256": "00" * 32}],
    )
    assert body["exit_code"] == 0
    observed = json.loads(artifact_path.read_text())
    assert observed["leak"] is False
    assert observed["home"].endswith("lane_cwd/.replay_lane_tmp/home")
    assert observed["home"] != str(poison_home)
    assert Path(observed["ceiling"]) == tmp_path


def test_git_ceiling_hides_parent_repo_from_lane_child(tmp_path: Path) -> None:
    """``git rev-parse`` from the lane cwd must not see a neighbour's repo."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    lane = tmp_path / "lane_cwd"
    lane.mkdir()
    _tracked(
        tmp_path,
        "lane_cwd/emit_git.py",
        "import json, subprocess\n"
        "from pathlib import Path\n"
        "p = Path('out')\n"
        "p.mkdir(parents=True, exist_ok=True)\n"
        "proc = subprocess.run(\n"
        "    ['git', 'rev-parse', '--is-inside-work-tree'],\n"
        "    capture_output=True, text=True, check=False,\n"
        ")\n"
        "(p / 'artifact.json').write_text(\n"
        "    json.dumps({'inside': proc.returncode == 0, 'out': proc.stdout.strip()}),\n"
        "    encoding='utf-8',\n"
        ")\n",
    )
    artifact_path = lane / "out" / "artifact.json"
    body = _replay(
        tmp_path,
        argv=["emit_git.py"],
        cwd="lane_cwd",
        artifacts=[{"path": "out/artifact.json", "sha256": "00" * 32}],
    )
    assert body["exit_code"] == 0
    observed = json.loads(artifact_path.read_text())
    assert observed["inside"] is False


def test_clean_replay_passes_control(tmp_path: Path) -> None:
    # control: unmutated carrier must pass — guards above are meaningful
    body = _replay(tmp_path)
    assert body["verdict"] == "pass"
    assert body["all_match"] is True
    assert body["exit_code"] == 0


@pytest.mark.parametrize("cwd", ["no_such_dir", "../replay_escape_probe"])
def test_refused_cwd_creates_no_directories(tmp_path: Path, cwd: str) -> None:
    """A receipt refused on its cwd must not choose where replay creates dirs."""
    root = tmp_path / "root"
    root.mkdir()
    body = _replay(root, cwd=cwd)
    assert body["verdict"] == "fail"
    assert str(body["spawn_error"]).startswith(("cwd_missing", "cwd_escapes_root"))
    assert not (root / "no_such_dir").exists()
    assert not (tmp_path / "replay_escape_probe").exists()
