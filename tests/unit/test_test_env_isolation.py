"""Process-env isolation: mutating test modules must not touch import time."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import GUARDED_ENV_BASELINE

_REPO = Path(__file__).resolve().parents[2]
_GUARDED = tuple(GUARDED_ENV_BASELINE)

_MUTATING_MODULES = (
    "tests.property.test_replay_determinism",
    "tests.unit.fx1_serve.test_idem_audit",
    "tests.unit.fx1_serve.test_batch_audit",
)


def _snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _GUARDED}


def test_guarded_env_unchanged_after_importing_mutating_modules() -> None:
    before = _snapshot()
    for name in _MUTATING_MODULES:
        sys.modules.pop(name, None)
        importlib.import_module(name)
    after = _snapshot()
    assert after == before, f"import-time env leak: before={before} after={after}"


def test_guarded_env_matches_session_baseline_after_collection() -> None:
    assert _snapshot() == GUARDED_ENV_BASELINE


def test_slow_marker_is_registered_in_pyproject() -> None:
    import tomllib

    data = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    assert any(str(entry).startswith("slow:") for entry in markers)


def test_seeded_threshold_probes_stable_across_openblas_kernels() -> None:
    targets = [
        "tests/unit/core/test_rank_confidence_sequences.py::test_bb_edge_margin_tau_shifts_null",
        "tests/unit/core/test_sliced_wasserstein.py::test_size_under_null_is_near_alpha",
    ]
    for coretype in ("HASWELL", "SKYLAKEX", "NEHALEM"):
        env = {**os.environ, "OPENBLAS_CORETYPE": coretype, "OPENBLAS_NUM_THREADS": "1"}
        for nodeid in targets:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", nodeid],
                cwd=_REPO,
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0, (
                f"OPENBLAS_CORETYPE={coretype} {nodeid} failed:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )


def test_strict_markers_collect_is_clean_on_isolation_modules() -> None:
    targets = [
        "tests/formal/test_stateful.py",
        "tests/property/test_replay_determinism.py",
        "tests/unit/fx1_serve/test_idem_audit.py",
        "tests/unit/fx1_serve/test_batch_audit.py",
        "tests/unit/test_hf_rv_scaffolding.py",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--strict-markers", "--collect-only", "-q", *targets],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
