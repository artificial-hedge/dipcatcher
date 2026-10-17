"""``make admission-gate`` must gate every receipt the epoch stamper counts.

The stamper (``corpus_epoch.member_digests``) is recursive: a ``*.json`` at any
depth under ``receipts/`` -- ``legacy-unsealed/``, a new subdirectory, a
non-ASCII name -- is a chain member. The Makefile once selected candidates with
``grep '^receipts/[^/]*\\.json$'`` over ``git diff --name-only``, so a new
nested receipt was stamped without ever passing ``admit-batch``, and git's
C-quoting of non-ASCII paths hid even flat ones.

Git paths only list the stamper's members while ``receipts/`` is a plain
directory tree that the working tree matches. A root symlink (``git mv receipts
zz_corpus && ln -s zz_corpus receipts``) puts new members outside the diff
pathspec, a nested symlink or submodule hides bytes, and an untracked or edited
file is stamped without being committed -- each must fail the gate. So must the
way ``git status`` itself goes blind locally: an assume-unchanged or
skip-worktree index flag.

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
    """corpus-relative candidate path -> verdict, from the admission receipts the run wrote."""
    out: dict[str, str] = {}
    for path in sorted((Path(env["RUNNER_TEMP"]) / "admission").glob("receipt_admission_*.json")):
        doc = json.loads(path.read_text())
        body = doc.get("payload", doc)
        out[str(body["candidate"])] = str(body["verdict"])
    return out


def _bodies(env: dict[str, str]) -> dict[str, dict[str, object]]:
    """corpus-relative candidate path -> admission receipt body."""
    out: dict[str, dict[str, object]] = {}
    for path in sorted((Path(env["RUNNER_TEMP"]) / "admission").glob("receipt_admission_*.json")):
        doc = json.loads(path.read_text())
        body = doc.get("payload", doc)
        out[str(body["candidate"])] = body
    return out


# Every one of these is a stamper member that no admission check may skip.
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
    expected = {rel.removeprefix("receipts/") for rel in HOSTILE}
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


def _git_add_commit(repo: Path, *paths: str) -> None:
    _git(repo, "add", "-f", "-A", "--", *paths)
    _git(repo, *GIT_ID, "commit", "-qm", "case")


def _reroot_corpus_behind_a_symlink(repo: Path) -> None:
    """``git mv receipts zz_corpus && ln -s zz_corpus receipts``, plus new members."""
    _git(repo, "mv", "receipts", "zz_corpus")
    (repo / "receipts").symlink_to("zz_corpus", target_is_directory=True)
    for rel in ("zz_corpus/zz_hostile.json", "zz_corpus/zz_sub/zz_hostile2.json"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(_sealed(UNSTAMPED))
    _git_add_commit(repo, "receipts", "zz_corpus")
    # The divergence this closes: the stamper follows the symlinked root and
    # counts the new members while the diff pathspec never lists them.
    members = member_digests(repo / "receipts")
    assert {"zz_hostile.json", "zz_sub/zz_hostile2.json"} <= set(members)


def _link_a_directory_into_the_corpus(repo: Path) -> None:
    (repo / "zz_outside").mkdir()
    (repo / "zz_outside" / "zz_hostile.json").write_bytes(_sealed(UNSTAMPED))
    (repo / "receipts" / "zz_link").symlink_to("../zz_outside", target_is_directory=True)
    _git_add_commit(repo, "receipts", "zz_outside")


def _link_a_file_into_the_corpus(repo: Path) -> None:
    (repo / "zz_target.json").write_bytes(_sealed(UNSTAMPED))
    (repo / "receipts" / "zz_link.json").symlink_to("../zz_target.json")
    _git_add_commit(repo, "receipts", "zz_target.json")


def _add_a_submodule_entry(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},receipts/zz_mod")
    _git(repo, *GIT_ID, "commit", "-qm", "case")


# Each re-shapes receipts/ so that ``git diff --name-only`` no longer lists what
# the stamper (which reads the filesystem) counts, or lists it as a non-file.
RESHAPED = {
    "root_symlink": _reroot_corpus_behind_a_symlink,
    "directory_symlink": _link_a_directory_into_the_corpus,
    "file_symlink": _link_a_file_into_the_corpus,
    "submodule": _add_a_submodule_entry,
}


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
@pytest.mark.parametrize("case", sorted(RESHAPED))
def test_a_corpus_that_is_not_a_plain_tree_fails_the_gate_instead_of_hiding_its_diff(
    gate_repo: Gate, case: str
) -> None:
    repo, base, env = gate_repo
    RESHAPED[case](repo)
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "no corpus receipt changes" not in out, out
    assert "plain directory tree" in out, out


@pytest.mark.parametrize("how", ["untracked", "modified"])
def test_a_working_tree_receipt_that_head_does_not_have_fails_the_gate(
    gate_repo: Gate, how: str
) -> None:
    """The stamper reads the working tree, the gate reads commits: a receipt the
    checkout holds but HEAD lacks would be stamped without being gated."""
    repo, base, env = gate_repo
    if how == "untracked":
        (repo / "receipts" / "zz_untracked.json").write_bytes(_sealed(UNSTAMPED))
        assert "zz_untracked.json" in member_digests(repo / "receipts")
    else:
        victim = sorted((repo / "receipts").glob("*.json"))[0]
        victim.write_bytes(victim.read_bytes() + b"\n")
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "no corpus receipt changes" not in out, out
    assert "differs from HEAD" in out, out


def _claim(value: float, *, inputs: str) -> bytes:
    """Sealed, honest receipt asserting one claim on a shared (private) dataset."""
    return _sealed(
        {
            **STAMPED,
            "schema": "synthetic_fixture.v1",
            "inputs_sha256": inputs * 32,
            "dataset_sha256": "5a" * 32,
            "params": {"drill": True},
            "results": [{"head": "x", "qlike": value}],
        }
    )


def test_a_nested_receipt_sharing_a_flat_name_is_still_checked_against_the_flat_one(
    gate_repo: Gate,
) -> None:
    """The admission lattice used to key candidates by file name: the second of
    ``zz_a.json`` / ``zzz/zz_a.json`` was compared against the first one's bytes,
    so a direct contradiction drew ``admit`` for both."""
    repo, base, env = gate_repo
    _commit(
        repo,
        {
            "receipts/zz_a.json": _claim(1.0, inputs="dd"),
            "receipts/zzz/zz_a.json": _claim(9.9, inputs="ee"),
        },
    )
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "verdict=quarantine" in out and "verdict=admit" in out, out
    assert _verdicts(env) == {"zz_a.json": "admit", "zzz/zz_a.json": "quarantine"}, out


def _porcelain(repo: Path) -> str:
    return _git(repo, "status", "--porcelain", "--untracked-files=all", "--", "receipts")


def _assume_unchanged_edit(repo: Path) -> None:
    """``git update-index --assume-unchanged`` on a member, then overwrite its bytes."""
    victim = sorted((repo / "receipts").glob("*.json"))[0]
    before = member_digests(repo / "receipts")[victim.name]
    _git(repo, "update-index", "--assume-unchanged", "--", victim.relative_to(repo).as_posix())
    victim.write_bytes(victim.read_bytes() + b"\n")
    # The divergence: the stamper counts new bytes while git status shows nothing.
    assert member_digests(repo / "receipts")[victim.name] != before
    assert _porcelain(repo) == ""


def _skip_worktree_intent_to_add(repo: Path) -> None:
    """``git add -N`` + ``--skip-worktree`` hides a brand-new member from git status."""
    hidden = repo / "receipts" / "zz_hidden_new.json"
    hidden.write_bytes(_sealed(UNSTAMPED))
    _git(repo, "add", "-N", "-f", "--", "receipts/zz_hidden_new.json")
    _git(repo, "update-index", "--skip-worktree", "--", "receipts/zz_hidden_new.json")
    assert "zz_hidden_new.json" in member_digests(repo / "receipts")
    assert _porcelain(repo) == ""


def _skip_worktree_edit(repo: Path) -> None:
    victim = sorted((repo / "receipts").glob("*.json"))[0]
    _git(repo, "update-index", "--skip-worktree", "--", victim.relative_to(repo).as_posix())
    victim.write_bytes(victim.read_bytes() + b"\n")
    assert _porcelain(repo) == ""


# Each leaves a stamper member that ``git status`` and the BASE..HEAD diff never list.
STATUS_BLIND = {
    "assume_unchanged_edit": (_assume_unchanged_edit, "cannot vouch for the working tree"),
    "skip_worktree_intent_to_add": (_skip_worktree_intent_to_add, "cannot vouch"),
    "skip_worktree_edit": (_skip_worktree_edit, "cannot vouch"),
}


@pytest.mark.parametrize("case", sorted(STATUS_BLIND))
def test_a_working_tree_git_status_cannot_see_fails_the_gate(gate_repo: Gate, case: str) -> None:
    """``git status --porcelain`` is blind to an assume-unchanged or skip-worktree
    flag, yet the stamper counts the file (it skips only untracked ignored ones):
    the gate once exited 0 ("no corpus receipt changes") over a changed or
    brand-new member."""
    repo, base, env = gate_repo
    setup, needle = STATUS_BLIND[case]
    setup(repo)
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "no corpus receipt changes" not in out, out
    assert needle in out, out


@pytest.mark.parametrize(
    ("slot", "verdict"),
    [
        ("receipts/legacy-unsealed/zz_claim_b.json", "reject"),
        ("receipts/legacy-unsealed/deep/zz_claim_b.json", "reject"),
        ("receipts/zz_vis/zz_claim_b.json", "quarantine"),
    ],
)
def test_a_contradicting_receipt_is_never_admitted_wherever_it_sits(
    gate_repo: Gate, slot: str, verdict: str
) -> None:
    """A sealed, honest receipt contradicting a corpus claim drew ``quarantine``
    when flat or in a subdirectory but ``admit`` under ``legacy-unsealed/``: the
    lattice skips that subtree, so its delta was measured against nothing. The
    gate must refuse it (outright, as ``suite-health`` does) and still report the
    contradiction the visible placements report."""
    repo, _head, env = gate_repo
    _commit(repo, {"receipts/zz_claim_a.json": _claim(1.0, inputs="dd")})
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {slot: _claim(9.9, inputs="ee")})
    proc = _gate(repo, base, env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    candidate = slot.removeprefix("receipts/")
    assert _verdicts(env) == {candidate: verdict}, out
    body = _bodies(env)[candidate]
    assert body["lattice_new_inconsistent"], body
