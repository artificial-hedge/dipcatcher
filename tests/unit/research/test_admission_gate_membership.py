"""``make admission-gate`` must gate every receipt the epoch stamper counts.

The stamper (``corpus_epoch.member_digests``) is recursive: a ``*.json`` at any
depth under ``receipts/`` -- ``legacy-unsealed/``, a new subdirectory, a
non-ASCII name -- is a chain member. The Makefile once selected candidates with
``grep '^receipts/[^/]*\\.json$'`` over ``git diff --name-only``, so a new
nested receipt was stamped without ever passing ``admit-batch``, and git's
C-quoting of non-ASCII paths hid even flat ones.

These tests run the REAL ``Makefile`` target and the production CLI (the same
``quant_fund.cli.main.app`` the ``dipcatcher`` console script runs) against the
real committed corpus, pins and quarantine registry, in a throwaway repo that
borrows this repo's objects: the repo under test is never modified. The only
stand-in is the ``uv`` launcher, which execs this interpreter instead of
resolving a virtualenv.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from quant_fund.research.corpus_epoch import member_digests
from quant_fund.utils.receipt import seal_receipt

REPO_ROOT = Path(__file__).resolve().parents[3]
GIT_ID = ("-c", "user.email=t@t", "-c", "user.name=t")

# Sealed and verifiable (verify-receipt accepts it) but without the honesty
# stamps admission demands -- the class only the admission gate refuses.
UNSTAMPED = {"kind": "synthetic_fixture.v1", "marker": "admission-gate-membership"}
# Fully stamped; what an honest new receipt looks like.
STAMPED = {
    "kind": "synthetic_fixture.v1",
    "marker": "admission-gate-membership-honest",
    "research_only": True,
    "live_pnl_claim": False,
    "data_label": "SYNTHETIC",
}
UNSEALED = b'{"schema": "bogus.v1", "research_only": false, "live_pnl_claim": true}\n'


def _sealed(body: Mapping[str, object]) -> bytes:
    return (json.dumps(seal_receipt(dict(body)), indent=2, sort_keys=True) + "\n").encode()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


Gate = tuple[Path, str, dict[str, str]]


@pytest.fixture
def gate_repo(tmp_path: Path) -> Gate:
    """(repo, base_sha, env): real Makefile, quality pins and receipts at HEAD."""
    if shutil.which("make") is None:
        pytest.skip("make is required")
    try:
        common = _git(REPO_ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")
        head = _git(REPO_ROOT, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("a git checkout of this repo is required")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    # Borrow the objects instead of copying them (a clone would be ~0.5 GB).
    (repo / ".git" / "objects" / "info" / "alternates").write_text(f"{common}/objects\n")
    _git(repo, "update-ref", "refs/heads/base", head)
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/base")
    _git(repo, "read-tree", "HEAD")
    _git(repo, "checkout", "HEAD", "--", ".gitignore", "quality", "receipts")
    # The Makefile under test is the working-tree one; left uncommitted so it
    # never appears in the BASE..HEAD diff the gate computes.
    shutil.copy2(REPO_ROOT / "Makefile", repo / "Makefile")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        '[ "$1" = run ] || exit 97\nshift\n'
        '[ "$1" = --frozen ] && shift\n'
        '[ "$1" = dipcatcher ] || exit 98\nshift\n'
        f'exec "{sys.executable}" -c "import sys; from quant_fund.cli.main import app; '
        'app(args=sys.argv[1:], prog_name=\'dipcatcher\')" "$@"\n'
    )
    uv.chmod(0o755)
    runner = tmp_path / "runner"
    runner.mkdir()
    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner),
    }
    return repo, head, env


def _commit(repo: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _git(repo, "add", "-f", "--", *files)
    _git(repo, *GIT_ID, "commit", "-qm", "case")


def _gate(repo: Path, base: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "admission-gate", f"ADMISSION_BASE={base}"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )


def _verdicts(env: dict[str, str]) -> dict[str, str]:
    """candidate basename -> verdict, from the admission receipts the run wrote."""
    out: dict[str, str] = {}
    for path in sorted((Path(env["RUNNER_TEMP"]) / "admission").glob("receipt_admission_*.json")):
        doc = json.loads(path.read_text())
        body = doc.get("payload", doc)
        out[str(body["candidate"])] = str(body["verdict"])
    return out


# Every one of these is a stamper member that no admission check may skip.
# Basenames are unique so each candidate is attributable in the batch.
HOSTILE = {
    "receipts/zz_flat_unsealed.json": UNSEALED,
    "receipts/zz_flat_sealed.json": _sealed(UNSTAMPED),
    "receipts/legacy-unsealed/zz_qdir_unsealed.json": UNSEALED,
    "receipts/legacy-unsealed/zz_qdir_sealed.json": _sealed(UNSTAMPED),
    "receipts/zz_sub/zz_sub_unsealed.json": UNSEALED,
    "receipts/zz_sub/zz_sub_sealed.json": _sealed(UNSTAMPED),
    "receipts/a/b/c/zz_deep_sealed.json": _sealed(UNSTAMPED),
    "receipts/café_zz_flat.json": UNSEALED,
    "receipts/zz_sub/café_zz_sub.json": _sealed(UNSTAMPED),
    # Membership is the platform's glob: Windows (the fleet box) matches
    # case-insensitively, Linux does not. The gate must not be narrower than
    # the widest platform, so it over-selects rather than skips.
    "receipts/zz_sub/zz_upper.JSON": _sealed(UNSTAMPED),
}


def test_no_stamper_member_in_the_diff_is_skipped_by_the_gate(gate_repo: Gate) -> None:
    repo, base, env = gate_repo
    _commit(repo, HOSTILE)
    members = set(member_digests(repo / "receipts"))
    # Precondition: the stamper really does count the nested and non-ASCII ones.
    for rel in HOSTILE:
        if not rel.endswith(".JSON"):  # case-sensitive on Linux, by design
            assert rel.removeprefix("receipts/") in members, rel
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "no corpus receipt changes" not in out, out
    verdicts = _verdicts(env)
    expected = {Path(rel).name for rel in HOSTILE}
    assert set(verdicts) == expected, (sorted(set(verdicts) ^ expected), out)
    assert "admit" not in set(verdicts.values()), verdicts


# Paths the gate cannot hand to ``admit-batch`` verbatim: git C-quotes a name
# with a double quote whatever ``core.quotePath`` says, and the unquoted
# ``$changed`` word list is split on whitespace and glob-expanded (``x[a].json``
# would silently become ``xa.json``). Each must fail the gate by design.
UNGATEABLE = {
    "double_quote": ({'receipts/zz_sub/zz"quoted.json': _sealed(UNSTAMPED)}, "git-quoted path"),
    "space": ({"receipts/zz space.json": _sealed(UNSTAMPED)}, "whitespace or glob"),
    "glob": (
        {
            "receipts/zz_g[a].json": _sealed(UNSTAMPED),
            "receipts/zz_ga.json": _sealed(UNSTAMPED),
        },
        "whitespace or glob",
    ),
}


@pytest.mark.skipif(sys.platform == "win32", reason="these characters cannot be file names")
@pytest.mark.parametrize("case", sorted(UNGATEABLE))
def test_a_path_the_shell_would_mangle_fails_the_gate_instead_of_being_skipped(
    gate_repo: Gate, case: str
) -> None:
    repo, base, env = gate_repo
    files, needle = UNGATEABLE[case]
    _commit(repo, files)
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "no corpus receipt changes" not in out, out
    assert needle in out and "rename it" in out, out


def test_gate_still_passes_when_nothing_in_receipts_changed(gate_repo: Gate) -> None:
    repo, base, env = gate_repo
    proc = _gate(repo, base, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no corpus receipt changes" in proc.stdout


@pytest.mark.slow
def test_gate_ignores_a_non_json_file_the_stamper_does_not_count(gate_repo: Gate) -> None:
    repo, base, env = gate_repo
    _commit(repo, {"receipts/legacy-unsealed/NOTE.md": b"not a receipt\n"})
    assert "legacy-unsealed/NOTE.md" not in member_digests(repo / "receipts")
    proc = _gate(repo, base, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no corpus receipt changes" in proc.stdout


@pytest.mark.slow
def test_honest_stamped_members_flat_nested_and_non_ascii_all_reach_admission(
    gate_repo: Gate,
) -> None:
    """Differential: every member the diff adds is in the admission batch, and
    honest, epoch-stamped receipts pass -- the widened filter must not
    over-block what the stamper allows."""
    repo, base, env = gate_repo
    before = set(member_digests(repo / "receipts"))
    files = {
        "receipts/zz_flat.json": _sealed(STAMPED),
        "receipts/zz_sub/zz_nested.json": _sealed({**STAMPED, "marker": "nested"}),
        "receipts/café_zz.json": _sealed({**STAMPED, "marker": "unicode"}),
    }
    _commit(repo, files)
    # Stamp with the production writer so the epoch chain is not the reason for
    # a refusal; the new epoch receipt is itself a flat candidate.
    subprocess.run(
        ["uv", "run", "--frozen", "dipcatcher", "corpus-epoch", "--corpus-dir", "receipts"]
        + ["--out-dir", "receipts", "--heads-pin", "quality/epoch_heads.json"],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    _git(repo, "add", "--", "receipts", "quality/epoch_heads.json")
    _git(repo, *GIT_ID, "commit", "-qm", "stamp")
    added = set(member_digests(repo / "receipts")) - before
    assert {rel.removeprefix("receipts/") for rel in files} <= added
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert f"admit-batch candidates={len(added)} " in out, out


@pytest.mark.slow
def test_restoring_the_registered_quarantined_records_is_not_blocked(gate_repo: Gate) -> None:
    """The widened filter newly routes ``legacy-unsealed/*.json`` through
    ``admit-batch``. Re-adding the registered records at their pinned bytes is
    a restoration of stamped history (``already_stamped``), not new evidence,
    and must still pass."""
    repo, head, env = gate_repo
    nested = sorted(
        p.relative_to(repo).as_posix()
        for p in (repo / "receipts" / "legacy-unsealed").glob("*.json")
    )
    assert nested, "the corpus registers quarantined records under legacy-unsealed/"
    _git(repo, "rm", "-q", "--", *nested)
    _git(repo, *GIT_ID, "commit", "-qm", "drop")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", head, "--", *nested)
    _git(repo, *GIT_ID, "commit", "-qm", "restore")
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert f"already_stamped={len(nested)} " in out, out
