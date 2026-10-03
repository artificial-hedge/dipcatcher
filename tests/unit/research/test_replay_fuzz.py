"""Adversarial mutations of replay carriers — every tamper class must fail closed.

The carrier seal is integrity, not authenticity: sha256 resealing is free, so
each mutation below is re-sealed honestly before the replay. The pins inside
``replay`` (argv, artifact digests, input tapes) are what must hold.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from quant_fund.research.replay_proof import run_replay


def _emit_script(root: Path) -> None:
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "emit.py").write_text(
        "import json, pathlib\n"
        "pathlib.Path('out/artifact.json').write_text(json.dumps({'fixed': True}))\n"
    )


def _carrier(root: Path, **replay_overrides: object) -> Path:
    artifact = b'{"fixed": true}'
    replay: dict[str, object] = {
        "argv": [sys.executable, "emit.py"],
        "artifacts": [
            {"path": "out/artifact.json", "sha256": hashlib.sha256(artifact).hexdigest()}
        ],
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
        artifacts=[{"path": "out/artifact.json", "sha256": "00" * 32}],
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
    (tmp_path / "noop.py").write_text("pass\n")
    body = _replay(tmp_path, argv=[sys.executable, "noop.py"])
    assert body["verdict"] == "fail"
    assert body["artifacts"][0]["note"] == "artifact_missing"


def test_stale_preplaced_artifact_does_not_satisfy(tmp_path: Path) -> None:
    # pre-place the pinned bytes; argv writes nothing -> stale copy must be
    # deleted pre-spawn, so the artifact is missing, not satisfied
    (tmp_path / "noop.py").write_text("pass\n")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "artifact.json").write_bytes(b'{"fixed": true}')
    body = _replay(tmp_path, argv=[sys.executable, "noop.py"])
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
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    tracked = tmp_path / "out" / "artifact.json"
    tracked.write_text("old committed bytes")
    subprocess.run(["git", "add", "out/artifact.json"], cwd=tmp_path, check=True)
    body = _replay(tmp_path)
    assert body["verdict"] == "fail"
    assert str(body["spawn_error"]).startswith("artifact_overwrites_committed")
    assert tracked.read_text() == "old committed bytes"


def test_nonzero_exit_fails(tmp_path: Path) -> None:
    (tmp_path / "die.py").write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path('out/artifact.json').write_text(json.dumps({'fixed': True}))\n"
        "sys.exit(3)\n"
    )
    body = _replay(tmp_path, argv=[sys.executable, "die.py"])
    assert body["verdict"] == "fail"
    assert body["exit_code"] == 3


def test_clean_replay_passes_control(tmp_path: Path) -> None:
    # control: unmutated carrier must pass — guards above are meaningful
    body = _replay(tmp_path)
    assert body["verdict"] == "pass"
    assert body["all_match"] is True
    assert body["exit_code"] == 0
