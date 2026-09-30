# Swarm playbook — running concurrent autonomous agents on this repo

Status: **extracted practice, not a validated method.** Every rule below comes
from one repository over **15 days** (2026-09-15 → 2026-09-29, `git log
--reverse` first commit → `git log -1` last commit). It is not proven across
repos, teams, or codebases. Where a rule is a judgment call rather than an
observed fact, it is marked **OPINION**.

Scope: how to run several autonomous coding agents (Devin, Cursor Agent,
opencode, Codex, dsh) against one working tree of hard scientific code without
them destroying each other's work or shipping unverifiable claims. This is the
*process* layer. It does not duplicate `docs/SOTA/11-testing-strategy.md`
(test adequacy), `docs/SOTA/19-code-quality.md` (code-quality findings),
`docs/SOTA/20-research-workflow.md` (study lifecycle), or
`docs/SOTA/21-concurrent-loop-profile.md` (forensic process/daemon inventory).
It cites them.

Honesty framing (AGENTS.md rules 1–4 apply to this document): nothing here is
market evidence. The repo's single real-data study returned verdict
**`deflated`** (`research/reality/studies/reality-us-liquid-daily-2026-09-27/receipt.json`:
`"verdict": "deflated"`, `n_trials: 29`, `dsr: 0.684`, `pbo: 0.122`). Volume of
agent-authored code is not evidence about markets. See §13.

---

## 0. How to read the citations

| Marker | Meaning |
|---|---|
| **OBSERVED** | Measured in-tree, or quoted from a tracked artifact, during this lane. The command or file:line is given. |
| **OPINION** | My judgment. Not established by any measurement in this repo. |
| *(no marker)* | Restatement of an existing repo contract (AGENTS.md, ADR, Makefile, CI YAML). |

**Measurement warning — read before trusting any number here.** This tree is
being written by other lanes while it is being measured. Over the ~2 h of this
lane, `git rev-parse --short HEAD` moved `57d305dd` → `a1f928ef` → `abc7a7d0`
(3 commits landed under a read-only documentation lane), and
`git status --porcelain | Measure-Object` returned **134 → 136 → 156**. The
divergence from `origin/main` moved from `922 24` to `934 27`
(`git rev-list --left-right --count origin/main...HEAD`).
`docs/SOTA/21-concurrent-loop-profile.md` §8b reached the same conclusion about
its own numbers: *"the entry count is a function of where in the 15 s push cycle
you sample it, not of durable repo state."* **OBSERVED.** Any count in this
document is a snapshot with a timestamp, not a property of the repo.

---

## 1. The shape of the swarm (measured)

`git shortlog -sn HEAD`, 2026-09-29, at `HEAD = 57d305dd`, 597 commits total:

| Author | Commits | Class |
|---|---|---|
| Advaith Vaithianathan | 340 | human |
| Devin AI | 154 | agent |
| Cursor Agent | 58 | agent |
| cursor-agent[bot] | 24 | agent (merge/PR bot identity) |
| devin-ai-integration[bot] | 12 | agent (merge/PR bot identity) |
| dependabot[bot] | 7 | bot |
| Codex | 1 | agent |
| cosmic-hydra | 1 | **same human** — `140935487+cosmic-hydra@users.noreply.github.com` is Advaith Vaithianathan's GitHub noreply handle (see `git show -s --format="%an %ae" 6c47b032`) |

Agent-attributable: **249 / 597 ≈ 42%** (256 / 597 ≈ 43% including dependabot).
At least three distinct agent products, plus a Mac-side sync daemon
(`docs/SOTA/21-concurrent-loop-profile.md` §5) and a 24×7 opencode loop
(§3.1), running concurrently against one directory.

Tracked Python at the same snapshot (`git ls-files "*.py"` + line count):
**1,871 files, 311,103 lines** — of which `src/*.py` 733 files / 162,419 lines
and `tests/*.py` 984 files / 115,077 lines. **OBSERVED.**

The claim this document exists to support is narrow: *that output was produced
without the agents permanently destroying each other's work, and without the
honesty gates being weakened to get it green.* Both halves are checkable
(§5, §10). It is **not** a claim that the code is correct (§13).

---

## 2. Lane structure: what a lane is and how to scope one

A **lane** is one agent session with (a) a written objective, (b) a claimed
identity, (c) a bounded file set, and (d) its own evidence directory. All four
are observable in this repo.

| Element | Where it lives | Evidence |
|---|---|---|
| Identity claim | `INFLIGHT` `owner:` field | `INFLIGHT:1` `owner: opencode (resumed session)`; `INFLIGHT:28` `owner: devin (research canon wave)` |
| Objective | `INFLIGHT` `slice:` / `status:` / `detail:` | `INFLIGHT:2-4`, `INFLIGHT:29-31` |
| Observed neighbours | `INFLIGHT` `Lane B (other session, observed):` | `INFLIGHT:25` |
| Per-lane artifacts | `.dsh-24x7/lane-*/` (22 dirs: `lane-aci`, `lane-egarch`, `lane-perf`, `lane-volm`, `lane-simlive`, `lane-xbeta`, …) | `Get-ChildItem .dsh-24x7 -Directory` |
| Per-lane gate logs | `.dsh-24x7/g-*.txt`, `gate-*.txt`, `lane-*/REPORT.md` | `.dsh-24x7/g-allow3.txt`, `.dsh-24x7/lane-perf/REPORT.md` |
| Lane test separation | `tests/fx1` outside default `testpaths` | `docs/adr/0003-fx1-test-lane-separation.md`; `pyproject.toml:317` |

**Scoping rules that worked (all OBSERVED in the artifacts above):**

1. **One bounded improvement per cycle, then stop.** The opencode 24×7 command
   definition is explicit: *"pick one bounded improvement, implement it, then
   verify … End after this one bounded cycle so an external supervisor can
   restart with fresh context"* and *"Never commit, push, delete the
   repository"*, *"Do not make broad speculative rewrites"*
   (`docs/SOTA/21-concurrent-loop-profile.md` §8c.4, quoting
   `.opencode/commands/24x7.md` in the `C:` clone).
2. **Name the file set up front and treat it as a contract.** `.dsh-24x7/lane-perf/REPORT.md`
   calls it *"the sanctioned file set"* and records the cost of obeying it (§4).
3. **Write evidence into a lane-private directory**, never into shared paths.
   The 22 `lane-*` dirs hold `.npz`/`.json` artifacts; the shared tree holds
   source.
4. **State what you did *not* do.** `INFLIGHT:204` *"Evals not wired into
   suite/cli — owner call."* `INFLIGHT:150-154` records a deferred CI perf gate
   *with the reason* ("local timings are not comparable across machines").
   `day_grind_progress.md` ends every wave with *"Next gap left on the table:"*.
5. **A read-only lane is a real lane.** `docs/SOTA/19-code-quality.md`,
   `docs/SOTA/11-testing-strategy.md`, `docs/SOTA/21-concurrent-loop-profile.md`
   and this file all declare "no source file was modified; this document is the
   only deliverable" and then attest to it (§9 of SOTA/21). Audit lanes produce
   findings that implementation lanes consume: SOTA/19 **P0-2** (honesty gates
   fail *open* on import error) was fixed the next day by
   `d87fe4a5 refactor(catalog): narrow lazy-import guards to ImportError`.

**OPINION.** The unit that scales is *lane-days*, not agents. Every wave entry
in `INFLIGHT` is a self-contained, gate-verified, uncommitted slice that a
human reviewed. Adding a sixth concurrent agent to a tree that already has five
does not add throughput if the file sets are not disjoint — it adds collision
surface (§4, §7).

---

## 3. The three prohibitions that make concurrency safe

These are the load-bearing rules. Each one exists because the alternative was
observed to fail.

### 3.1 Disjoint file allowlist

**Rule.** Each concurrent agent gets an explicit list of files it may create or
modify. Overlap between two live lanes is a planning bug, resolved *before* the
agents start, not by merge tools afterwards.

**Evidence that the rule is real and enforced:**

- `.dsh-24x7/lane-perf/REPORT.md` §"Scope note": an earlier parallel iteration
  had extended `engine.py::_build_result` and kept the kernel in a separate
  `_fast_kernel.py`. *"Both exceeded the sanctioned file set, so the engine
  change was reverted (git-clean now) and the kernel inlined into
  `fast_replay.py`. Cost of compliance: ~0 ms for the inline (perf-neutral) and
  ~2–4 ms for row-dict emission vs frame pass-through."* The lane also reports
  the honest residual it could not touch: *"`_build_result` (~42 ms, 64% of
  run): shared metrics tail; **off-limits (engine.py)**."* **OBSERVED.**
- The header of the same report states the objective as *"No changes to
  `engine.py`, `risk_gate.py`, or any parity-defining test."* **OBSERVED.**
- `.dsh-24x7/HANDOFF.md:127-129` records the inverse case — a lane that
  *observed* another agent's territory and left it alone: *"ANOTHER AGENT ACTIVE
  in this worktree: new `lane-*` dirs … + WIP edits to `engine.py` /
  `fast_replay.py` / `splice_challenger_column.py` / `robinhood_plus/constants.py`
  left UNCOMMITTED — **do not clobber**."* **OBSERVED.**

**Why it works.** A disjoint allowlist makes the write set of the swarm
*statically known*. Two agents that cannot touch the same file cannot produce a
content conflict, cannot silently revert each other, and cannot make the other's
gate results non-reproducible. It also converts "who broke this?" from an
archaeological question into a lookup.

**What it does not prevent — OBSERVED, and the most instructive failure in this
lane.** Disjoint *files* do not imply disjoint *concepts*. On 2026-09-27 two
different agents shipped two wave-9 module sets:

| Concept | Implementation A | Implementation B |
|---|---|---|
| Quantile regression forest | `src/quant_fund/models/qrf.py` → `class QuantileRegressionForest` (`:79`), committed `ffaf1f76` by `devin-ai-integration[bot]` | `src/quant_fund/models/quantile_forest.py` → `class QuantileRegressionForest` (`:90`), committed `6c47b032` by Advaith Vaithianathan |
| WATCH martingale monitor | `src/quant_fund/metrics/watch.py` (`WATCHMonitor`, `WATCHStep`) `3c7774e6`; `src/quant_fund/metrics/conformal_martingale.py` (`WatchMonitor`, `:210`) | `src/quant_fund/models/watch.py` (`WatchMartingale`, `:367`) `6c47b032` |
| EnbPI | `src/quant_fund/models/enbpi.py` (`class EnbPI`, `:106`) | — |

All six modules and their four test files (`tests/unit/models/test_qrf.py`,
`tests/unit/models/test_quantile_forest.py`, `tests/unit/models/test_watch.py`,
`tests/unit/core/test_watch.py`) are **tracked** (`git ls-files` confirms each).
Only one QRF is wired into the research battery: `research/benches_w810.py:24`
imports `from quant_fund.models.qrf import QuantileRegressionForest`, so
`models/quantile_forest.py` is exercised only by its own test file. `INFLIGHT`
carries **two separate "canon wave 9" entries** (`:109` and `:155`) describing
overlapping module lists — the record shows the duplication was never noticed as
duplication. **OBSERVED.**

**OPINION.** The allowlist must be a *concept* allowlist as well as a file
allowlist: "wave 9 = EnbPI + QRF + NGBoost + conformal martingales" should be
claimable by exactly one lane, the same way `H46–H51` was (§6). A module-name
registry (or a `grep` for the class name before starting) is the cheap fix.

### 3.2 Git-command prohibition set + single committer

**Rule.** Concurrent agents are forbidden from *all* state-mutating git
commands. One orchestrator performs commits serially afterwards.

Forbidden set (as applied to this lane, and as the correct default for any
concurrent lane):
`git add`, `git commit`, `git stash`, `git checkout`, `git restore`,
`git reset`, `git clean`, `git merge`, `git rebase`, `git pull`, `git push`,
`git switch`.
Allowed and encouraged: `git log`, `git show`, `git status`, `git diff`,
`git ls-files`, `git shortlog`, `git cat-file`, `git rev-parse`,
`git rev-list`, `git check-ignore`, `git stash list`, `git branch -a`.

**Why every one of those twelve is on the list.** Each mutates one of exactly
three shared resources — the index, `HEAD`, or the working tree — and all other
lanes read those resources to decide what to do and to compute their gate
results.

**Evidence the prohibition is the house rule, not an invention of this lane:**

- `INFLIGHT` records **"No commits."** / **"no commits made."** at lines
  `26`, `66`, `108`, `154`, `204`, `225` — i.e. six of the wave entries
  explicitly end with work left uncommitted for the owner. `INFLIGHT:1` names
  the owner as `opencode (resumed session)`, so the convention survives session
  restarts.
- `docs/SOTA_CANON_ROADMAP_2026_09.md` §4 states the acceptance rule for a wave:
  *"ruff + ruff-format + mypy clean on touched files; **no commits (repo
  convention: owner reviews)**."* **OBSERVED.**
- `docs/SOTA/21-concurrent-loop-profile.md` §9 attests to the read-only
  discipline for an investigation lane: *"Committed nothing. No `git add`,
  `commit`, `stash`, `checkout`, `reset`, or `clean`."*
- `.opencode/commands/24x7.md` hard boundaries: *"Never commit, push, delete the
  repository"* (quoted in SOTA/21 §8c.4).
- AGENTS.md keeps the *human/orchestrator* rule separate: *"Commits land
  directly on `main` for fx-1 lanes; cross-cutting repo changes go through
  PRs."* The single-committer pattern is what makes that safe under concurrency.

**Corollary — a conflicted `stash pop` does not drop the stash, so leave it.**
`git stash list` in this tree still returns `stash@{0}: On main:
pre-sync-20260928`, and `.dsh-24x7/HANDOFF.md:200` documents the same stash
(*"Uncommitted working-tree version preserved verbatim during the origin/main
sync (was stashed as `pre-sync-20260928`)"*). **OBSERVED, 2026-09-29.** The
recoverability of the other side of a conflict is worth more than a clean
`git stash list`.

**Corollary — back up before touching a conflicted file.** A scratch copy
outside the repo costs nothing and is the only recovery path when the file is
also untracked (see §5). `.dsh-24x7/pre-restore-status.txt` (103 lines) and the
restore agent's `.backup-prerestore/` snapshot (SOTA/21 §6.1) are this pattern
in the wild.

### 3.3 No repo-wide formatters or fixers mid-flight

**Rule.** Never run `ruff format src tests`, `ruff check --fix src tests`,
`make fmt`, or any whole-tree rewrite while other lanes have uncommitted work.
Format only the files in your allowlist.

**Why.** The rewrite set is not your change set.

- `make fmt` is literally `ruff check --fix src tests` + `ruff format src tests`
  (`Makefile:32-35`). The lint scope is `src` + `tests` = **1,717 tracked `.py`
  files** (`git ls-files "src/*.py" "tests/*.py" | Measure-Object`). A
  repo-wide format therefore touches 1,717 files including every other lane's
  in-flight work. **OBSERVED** (count measured this lane; the ~1,720 figure in
  the orchestrator's brief matches).
- Pre-existing format debt makes this worse: `INFLIGHT:70-72` records
  *"Pre-existing repo-wide format debt (~80 files) and `tests/tests` leftover
  tree noted, untouched beyond autofixed import sorting."* A repo-wide format
  would have produced an ~80-file diff attributed to a canon-wave lane.
- The correct scoped form is already the house style: `INFLIGHT:65-66`
  *"verification: ruff check clean; **new files** ruff-formatted; mypy clean on
  all new modules"*; `docs/SOTA_CANON_ROADMAP_2026_09.md` §4 *"clean **on
  touched files**"*.
- The no-clobber instinct is documented a week before this lane:
  `.dsh-24x7/HANDOFF.md:145-147` *"Pre-existing lint noise (not this session's)
  … `fast_replay.py` needs `ruff format` (concurrent agent's uncommitted WIP —
  **flagged, not clobbered**)."* Same note at `.dsh-24x7/PROGRESS.md:551-552`.
  **OBSERVED.**

**Interaction with pre-commit.** `.pre-commit-config.yaml` runs `ruff --fix` and
`ruff-format` (file-scoped, via pre-commit's staged-file list — safe) but also
two hooks with `pass_filenames: false`: `uv run mypy src/quant_fund` and
`uv run mypy src/fx1` — i.e. **every commit triggers a repo-wide type check**,
which will surface other lanes' in-flight type errors as your commit failure.
`uv-lock-check` (`uv lock --check`) likewise fails on another lane's
`pyproject.toml` edit. **OBSERVED** (config read this lane). **OPINION:** this
is acceptable *because* of §3.2 — only the orchestrator commits, so the
repo-wide hooks fire in a serial context, not inside five concurrent agents.

---

## 4. Lane checklist (copy-paste)

```text
LANE <name>  owner: <agent product + session id>
slice:      <one bounded improvement>
allowlist:  <explicit file list; NO overlap with a live lane>
concept:    <the module/class/hypothesis names this lane owns>
forbidden:  all state-mutating git (§3.2); repo-wide formatters (§3.3)
verify:     scoped gates only (§8)
evidence:   .dsh-24x7/lane-<name>/   (artifacts, gate logs, REPORT.md)
report:     append to INFLIGHT — owner/slice/status/detail/note
            state what was NOT done and why
commit:     NONE. Orchestrator commits serially.
```

---

## 5. Untracked work is the real risk — the PURGE INCIDENT

### 5.1 The incident

`INFLIGHT:205-209`, verbatim:

> PURGE INCIDENT 2026-09-27: parallel lane `chore/purge-derived-history`
> hard-reset main to a rewritten origin/main; all wave 9+/eng/fx-1 files were
> untracked and briefly vanished, then restored byte-identical (stash-pop).
> Wave 8 (committed) was carried into the rewritten history.
> INFLIGHT/reference entries re-appended.
> **Commit promptly — untracked files remain at risk.**

What the record lets us verify independently, **OBSERVED** this lane:

| Claim | Verification |
|---|---|
| The lane existed | `git branch -a` lists `remotes/origin/chore/purge-derived-history`; its tip is `6ebb312d` (2026-09-27 13:10 +0530) |
| Its purpose was history rewriting of derived data | Same branch carries `5c988bfc chore: untrack derived blobs (file_us_wide, 24x7 eval-full/paper_data) per AGENTS.md` |
| Wave 8 was committed and survived | `metrics/energy_score.py`, `metrics/anytime_fdr.py`, `metrics/e_detectors.py`, `models/posthoc_calibration.py`, `models/agaci.py` all tracked; committed on that branch at `a6f178d5 metrics/models: energy score, anytime fdr + e-detectors, agaci, posthoc calibration + tests` |
| Waves 9–10 were eventually committed | `models/enbpi.py`, `models/qrf.py`, `models/ngboost_lite.py`, `metrics/watch.py`, `metrics/conformal_martingale.py`, `validation/regime_eval.py`, `validation/leakage_redteam.py`, `research/benches_w810.py` all tracked |
| Something is *still* untracked and still at risk | `git status --porcelain -- src/quant_fund/backtest/_fast_kernel.py` → `?? src/quant_fund/backtest/_fast_kernel.py` (a 600-line numba kernel with a bit-identity contract, documented in `docs/SOTA/13-performance.md:176` and `.dsh-24x7/PROOF.md:48`) |

The last row is the point. **Two days after an incident whose stated lesson was
"commit promptly — untracked files remain at risk", a load-bearing kernel is
still untracked**, and is imported by two root-level probe scripts
(`probe_kernel.py:10`, `run_fallback_tests.py:5`) that are themselves loose
files in the repo root. A `git clean -xfd` by any lane would delete it with no
recovery path.

### 5.2 Why "no commits" and "commit promptly" coexist

They are not contradictory; they are two rules for two different actors.

- **Agent:** do not commit (§3.2). Your work stays in the tree, verified, for
  owner review — `docs/SOTA_CANON_ROADMAP_2026_09.md` §4.
- **Orchestrator:** commit promptly. The window between "agent finished" and
  "orchestrator committed" is the *only* window in which a `reset`/`clean`/
  `checkout`/sync-daemon write can destroy work with no git object to recover
  it from.

**OBSERVED cost of leaving that window open:** the purge lane's `reset` erased
wave 9+ files and recovery depended on a stash that happened to exist. The
`.dsh-24x7/HANDOFF.md:200` stash (`pre-sync-20260928`) is a second instance of
the same rescue. Two rescues in three days is a pattern, not luck.

**OPINION.** The mitigation is not "agents may commit" — that breaks §3.2. It
is: (a) orchestrator commits on a short cycle, (b) each lane writes a
`git bundle` or a plain directory copy of its allowlist into
`.dsh-24x7/lane-<name>/` as it goes, and (c) `git reset --hard` / `git clean`
are forbidden repo-wide, not just inside lanes, whenever any lane is live. A
history-rewrite lane must be scheduled as a **quiesce window**: no other lane
running, tree committed, then rewrite.

### 5.3 The moving-target corollary

A status count is not a state. Measured this lane: 134 → 136 → 156 entries over
~2 h; SOTA/21 measured 103 (baseline) → 135 → 140 → 146 → 123 over ~40 minutes
and concluded the number depends on where in the daemon's 15 s push cycle you
sample. **Do not use a `git status` count as a before/after invariant for a
lane.** Use per-file hashes on the allowlist instead:
`git hash-object <file>` vs `git rev-parse HEAD:<file>` — the technique SOTA/21
§6.2 used after concluding *"Do not trust `git diff` alone mid-churn; the index
stat cache and concurrent `git stash` make it unreliable here."*

---

## 6. Namespace collision avoidance: the H-id protocol

Research hypotheses in this repo carry stable IDs (`H2`, `H23`–`H51`, `H99`)
that receipts, catalog consistency gates, and `verify-research` all key on. Two
concurrent lanes inventing hypotheses will collide on the ID space. This repo
has a working protocol, and it is documented in the record.

**The incident — `INFLIGHT:23-25`, verbatim:**

> H-id note: H46–H51 claimed by the concurrent northset lane mid-build; the
> ranking data-snooping hypothesis is **H99** (documented headroom).
> Lane B (other session, observed): northset cluster-robust inference + H46–H51.

**Independent verification in the shipped catalog** (`src/quant_fund/research/catalog/hypotheses.py`),
**OBSERVED**:

| Line | Content | Meaning |
|---|---|---|
| `:991-996` | `H46_HYPOTHESIS_ID = "H46_northset_reject_liq_control"` … `H51_HYPOTHESIS_ID = "H51_northset_follow_overnight_gap"` | The northset lane **won** H46–H51 |
| `:274-275` | `H99_HYPOTHESIS_ID = "H99_ranking_data_snooping"` / `H99_EXPECTED_FAMILY = "discovery"` | The ranking lane **vacated** to reserved headroom |
| `:1319-1334` | both sets exported in `__all__` | Both claims are live in the same module |

So the collision happened, was detected mid-build, and was resolved by one lane
relocating rather than by either lane overwriting the other.

**The protocol, extracted:**

1. **Reserve headroom in the ID space.** H99 exists as "documented headroom" —
   a slot that is not the next sequential number, so a lane that is bumped has
   somewhere to go without renumbering anyone. **OPINION:** reserve a block
   (H90–H99) rather than a single value; the ranking lane got lucky that H99
   was free.
2. **Record the claim where the other lane will read it.** The claim lives in
   `INFLIGHT` next to the work, not in a private channel. `INFLIGHT:25` even
   records the *other* lane's scope as observed (`Lane B (other session,
   observed)`), which is how the collision was noticed at all.
3. **Move, don't fight.** The displaced lane took the reserved slot and
   re-labelled its own artifacts. Nothing was merged, nothing was overwritten.
4. **Sweep the prose after the move — this step was skipped.** The ID moved to
   H99 but the surrounding text still says H46: `hypotheses.py:270` *"the
   notebook must mint ``H46_ranking_data_snooping`` (discovery)"*,
   `:296` docstring *"True iff hypotheses contains ``H46_ranking_data_snooping``"*,
   `:308` and `:313` the same, while the actual error string at `:320` is
   `hypothesis_h99_missing_despite_finite_spa_p`. **OBSERVED.** Four stale
   references to a claim that no longer exists. **OPINION:** a collision
   resolution is not finished until `git grep H46_ranking` returns nothing.

**Generalization (OPINION).** Any shared, human-allocated identifier space needs
this protocol: hypothesis IDs, benchmark family names
(`catalog/registry.py:66` `"regime_eval"`, and `INFLIGHT:218-219` records
`BENCHMARK_FAMILY_ORDER untouched per the docs-consistency contract`), ADR
numbers, receipt IDs, PR branch names (`devin/1790496175-wave9-conformal-ts-trees`,
`INFLIGHT:110`). The mechanism is always: reserve headroom → publish the claim →
move on collision → sweep.

---

## 7. Gate discipline: scoped beats repo-wide mid-flight

The gates (AGENTS.md, `Makefile`):

| Gate | Command | Scope | Safe to run mid-flight? |
|---|---|---|---|
| Lint | `make lint` → `ruff check src tests` + `ruff format --check src tests` + `scripts/check_mypy_strict_allowlist.py` | repo-wide, **read-only** | Yes (read-only), but the *result* is contaminated by other lanes |
| Types | `make typecheck` → `mypy src/quant_fund` + `mypy --strict` on the public facade | repo-wide, read-only | Yes, same caveat |
| Lab tests | `make test` → `pytest -n auto --dist loadfile -m "not network and not slow"` | `testpaths` = 5 dirs | **No** — see below |
| Full lab | `make test-full` (adds `slow`) | same | **No** |
| fx1 suite | `make fx1-test` → `PYTHONPATH=src pytest tests/fx1 -q` | `tests/fx1` only | Yes, lane-scoped by design (ADR-0003) |
| fx1 full gate | `make fx1-gate` = `fx1-lint` + `mypy src/fx1` + `tests/fx1` + `-k honesty` + corpus smoke | fx1 only | Yes |
| Format (destructive) | `make fmt` | 1,717 files | **Never mid-flight** (§3.3) |
| CI parity | `make sync` → `uv sync --frozen --all-groups --all-extras` | mutates `.venv` | **No** — one shared venv |

**Why per-lane scoped verification beats a repo-wide run mid-flight:**

1. **A repo-wide result is unattributable.** With 136–156 dirty entries from
   other lanes (`git status --porcelain`, measured this lane), a `make test`
   failure cannot be assigned to your change. `docs/SOTA/11-testing-strategy.md`
   §4.6 hit exactly this and had to declare its own finding blocking: the
   working-tree `pyproject.toml` reverted to a variant that dropped 3 of 6
   registered markers, so `--strict-markers` **failed collection** —
   *"Verified: `pytest tests/property/test_native_kernels.py --collect-only` →
   ERROR … 'native' not found in `markers` configuration option"* — and
   concluded *"the working tree cannot currently pass the CI it describes."*
   That was another lane's edit, and it invalidated every suite-wide number in
   the audit.
2. **Collection itself is unsafe with mirrors present.**
   `.dsh-24x7/HANDOFF.md` / `PROGRESS.md:72`: *"An earlier broad pytest
   invocation with `testpaths = [tests]` failed with **454 duplicate-module
   collection mismatches** from the tracked `tests/tests` mirror; that output is
   invalid after the discovery fix."* The durable control is the pinned
   `testpaths` list (`pyproject.toml:317`, 5 explicit dirs, never bare `tests`).
3. **One shared virtualenv is a serialization point.** `uv sync` mutates
   `.venv`; two lanes syncing different dependency sets corrupt each other.
   `.dsh-24x7/lane-perf/REPORT.md` §Reproduce shows the workaround actually
   used: a separate scratch checkout (`D:\dipcatcher-megaplan\_lane-perf`,
   `data/` junctioned) and a separate interpreter
   (`D:/bench-qlib/Scripts/python.exe`) for benchmark work.
4. **Scoped runs are what the record actually contains.** Every `INFLIGHT` wave
   reports *focused* counts: `42 new` (`:16`), `~257 tests green remotely`
   (`:64`), `126 tests` (`:77`), `108 tests green (7+12+40+27+22)` (`:107`),
   `39 focused tests green` (`:120`), `16 focused WATCH tests green` (`:136`),
   `65 tests green` (`:156`), `27 tests green` (`:170`), `22 tests green`
   (`:190`), `26 tests green` (`:198`), `9 tests` + `188 tests` (`:220-225`).
   Plus per-lane gate logs: `.dsh-24x7/g-pytest-lane1.txt` (17 KB),
   `g-pytest-research-serial.txt`, `g-mypy2.txt`, `g-ruff.txt`, `g-allow3.txt`.
   **OBSERVED.**

**Recommended per-lane verification set** (**OPINION**, built from the above):

```bash
# read-only, allowlist-scoped — safe with other lanes live
uv run ruff check <allowlist>
uv run ruff format --check <allowlist>
uv run mypy <allowlist modules>
uv run pytest <the test files for this lane> -q -p no:randomly
uv run python -c "import <each touched module>"     # cheapest real signal
# provenance, not git state:
git hash-object <allowlist> ; git rev-parse HEAD:<path>
```

Full gates (`make lint`, `make typecheck`, `make test`, `make fx1-gate`,
`uv sync --frozen`) belong to the **orchestrator**, run after the tree is quiet
and committed — which is exactly the AGENTS.md framing ("Gates (run before
committing)") and the CI framing (`.github/workflows/ci.yml`: lint / audit /
package / container / test matrix py3.12+3.13 × 4 shards / coverage / perf /
smoke / mc-engine-smoke / rust-accel / market-sim / parity-smoke / examples /
audit-observability / diffbacktest / robustness-smoke / formal / pretrade-risk /
stress-smoke; `.github/workflows/fx1.yml`: lint / mypy / tests / honesty
inheritance / corpus smoke). CI is the only place where the environment is
guaranteed single-owner.

**CI parity detail worth knowing:** AGENTS.md says `make sync` includes the
torch `nn` extra "matching the CI test job". True for `test` (`uv sync --frozen
--all-groups --all-extras`, `ci.yml:198`), but the `lint` job syncs
`--all-groups` **without** `--all-extras` (`ci.yml:52`), as do `smoke`,
`formal`, `mc-engine-smoke`, `audit-observability`, `rust-accel`. A local
`make sync` is therefore a *superset* of most CI jobs — a lint pass locally is
stronger evidence than a lint pass in CI, not weaker. **OBSERVED.**

---

## 8. The Honesty contract as an agent-output constraint

AGENTS.md's four hard rules are the reason agent-authored research code can be
trusted at all: they are enforced by *code that scans artifacts*, not by review
(`docs/adr/0008-forbidden-metric-key-scan.md`: *"Honesty is enforced by
scanning artifact keys, not by convention"*).

| Rule | Enforcement point | Verified location |
|---|---|---|
| Proper scores only; never headline Sharpe/Sortino/Calmar/P&L/NAV | `FORBIDDEN_RESEARCH_METRIC_KEYS = {sharpe, sortino, calmar, pnl, nav}` | `src/quant_fund/research/catalog/registry.py:101-109` |
| Same set mirrored into the model | `FORBIDDEN_HEADLINE_TOKENS = frozenset({"sharpe","sortino","calmar","pnl","nav"})` + alias table (`p&l`, `net asset value`, homoglyph fold) | `src/fx1/honesty.py:42-53`; drift blocked by `tests/fx1/test_honesty_inheritance.py` (blocking step in `fx1.yml`) |
| SYNTHETIC results are correctness tests, never market evidence | CI smoke asserts `notebook["data_source"] == "SYNTHETIC"` and `research_only is True`, `live_pnl_claim is False` | `.github/workflows/ci.yml:385-431`; `ci.yml:486-497` for mc-engine |
| No live-trading claims | `RuntimeConfig.live_must_be_explicit` raises `allow_live is unsupported: no live broker adapter in this repository`; CI greps `live_allowed: False` | `docs/SOTA/19-code-quality.md` §P3-1; `ci.yml:368-369`; five minimum-evidence conditions in `docs/INSTITUTIONAL_READINESS.md` |
| Receipts are immutable evidence | `receipt.v2` seal, `receipt_sha256`, `code_sha256`, `live_pnl_claim: Literal[False]`, `dipcatcher verify-research`, `make receipts-reverify` | `docs/adr/0001-immutable-self-sealing-receipts.md`; `Makefile:187-188` |
| Forbidden metrics absent per family | CI requires `forbidden_metrics_absent` true for **every** required family; stress-smoke fails on `grep -E 'sharpe:|sortino:|calmar:'` | `ci.yml:433-441`; `ci.yml:737-740` |

**How this constrains agent output in practice, OBSERVED:**

- Wave entries self-label. `INFLIGHT:108` *"Research/infrastructure only — no
  live broker / vendor MD / live_pnl_claim."* `INFLIGHT:120`, `:136`, `:180-181`
  *"SYNTHETIC-only assertions"*, *"All artifacts stamped SYNTHETIC /
  research_only"*. Every `day_grind_progress.md` wave ends with
  *"Research/infrastructure only - no live broker / vendor MD / live_pnl_claim"*.
- Audit lanes restate it as a framing obligation. `docs/SOTA/11-testing-strategy.md`
  header: *"Nothing in it is a market result, a performance claim, or evidence
  of profitability."* `docs/SOTA/24-metrics-findings.md:9-14` same, and names
  the admissible score list.
- **A fail-open hole in the honesty layer was found by an agent and fixed by
  another.** `docs/SOTA/19-code-quality.md` **P0-2**: four catalog soft-verify
  functions returned `[]` (= "no errors") when a deferred import raised, so
  *"An empty error list is indistinguishable from a passing check"* — in a
  system whose value proposition is fail-closed verification. Fixed by
  `d87fe4a5 refactor(catalog): narrow lazy-import guards to ImportError;
  tighten ratchet ceiling to 72` (2026-09-28), touching
  `research/catalog/candle.py`, `research/catalog/receipt.py`,
  `tests/unit/test_quality_ratchet.py`. **OBSERVED.**
- **The contract does not catch everything, and the record says so.**
  `docs/adr/0008` §Consequences: *"Renaming a metric to dodge the token list
  (`s_h_a_r_p_e` splits to different tokens) is only partially covered — a known
  residual risk."* Values are never scanned, only keys and headline patterns.
  **OPINION:** an agent that wants to smuggle a Sharpe claim can still do it in
  prose; the gate stops the artifact, not the narrative. That is why §9 (verify
  the claim) is not optional.

**Practical instruction to an agent (OPINION, derived from the above):** if your
lane produces a number, state (a) which proper score it is, (b) SYNTHETIC or
REAL, (c) what it does *not* establish. The wave entries that do this are the
ones that survived review unchanged.

---

## 9. Verify the claim, not the report

An agent's self-report is a hypothesis. This repo's culture is receipts over
assertions (AGENTS.md rule 4), and the same standard applies to the agents
themselves. Four worked examples, all **OBSERVED** this lane.

### 9.1 A lint error that was not a defect (`cpcv.py`)

Reported during this session: `src/quant_fund/validation/cpcv.py` had an
agent-visible "unused import" (F401) on `dataclass`.

Verification: `from dataclasses import dataclass` is at **line 10** and
`@dataclass(frozen=True)` is at **line 370** (`grep -n dataclass
src/quant_fund/validation/cpcv.py`). The import is used. The F401 was an
artifact of a **conflicted working tree**: with `<<<<<<<` / `=======` /
`>>>>>>>` markers in the file, the region containing line 370 is not parseable
as the same module, so the usage disappears from the tool's view.

**Lesson.** A tool error on a dirty tree is evidence about the tree, not about
the code. Before believing any lint/type/test result, check
`git status --porcelain -- <that file>` and
`grep -n '^\(<<<<<<<\|=======\|>>>>>>>\)' <that file>`. This is why
`.pre-commit-config.yaml` carries `check-merge-conflict`, and why a repo-wide
`ruff check` run mid-flight is unattributable (§7).

### 9.2 A "corrected citation" that is inconsistent in tree

`INFLIGHT:171-173` (canon wave 10): *"models/watch.py: WatchMartingale —
weighted-conformal test martingales (Prinster, Han & Saria 2025,
arXiv:2505.04608; **recon report's author list was wrong — corrected in
citation**)"*.

Verification: the two in-tree WATCH modules disagree about that author list.

| File | Line | Citation |
|---|---|---|
| `src/quant_fund/models/watch.py` | `:9-10` | *"Prinster, Han & Saria (2025, … arXiv:2505.04608, ICML)"* — 3 authors |
| `src/quant_fund/metrics/watch.py` | `:3-5` | *"Prinster, Han, Liu & Saria (2025), … ICML, PMLR 267 (arXiv:2505.04608)"* — 4 authors |

The self-correction is genuine and is the most instructive kind of entry in
`INFLIGHT` — an agent checked a secondary source against the paper and fixed
it. But it landed in one of two duplicate implementations (§3.1) and the other
still carries the short list. **Lesson:** a correction is not complete until
`git grep` for the wrong form returns nothing. Self-correction *records* are
reliable here; self-correction *propagation* is not.

### 9.3 A flagged math deviation that never reached the code

`INFLIGHT:194-196` (fx-1 capability lane wave 1): *"Flagged math deviation:
miscalibrated oracle caps ECE ~0.05 → default `ece_threshold=0.05` cannot fail
it; lower to ~0.02 for a harder gate."*

Verification: `src/fx1/eval/calibration_eval.py:360` still reads
`ece_threshold: float = 0.05`, and the pass condition at `:421-423` is
`math.isfinite(ece) and ece <= ece_threshold and …`. The flagged weakness is
**un-actioned in code**. **OBSERVED.**

Similarly, `INFLIGHT:164-166` documents EnbPI's *"adaptive width correction
(phi_t; documented deviation: papers define no phi recursion — transplanted
from ACI)"*. `grep -i "phi_t|deviation|transplant"` over
`src/quant_fund/models/enbpi.py` returns **nothing**; the shipped module
describes only the paper's Algorithm 1 β-minimising band (`:14-18`, `:78-81`).
The deviation note exists in `INFLIGHT` and not in the code that would have
carried it — because a *different* EnbPI shipped (§3.1).

**Lesson (the important one).** A deviation-from-source note that lives only in
a work-tracking file is not a durable record. It is invisible to a reader of
the module, to `verify-research`, and to CI. **OPINION:** math deviations
belong in the module docstring *and* in `docs/MATH_SPEC.md` (which
`.gitattributes` gives `merge=union` precisely so concurrent lanes can both
append to it), the same way
`docs/FX1_TRAINING.md` uses greppable `**Status: PLANNED**` / `PARTIAL` /
`SHIPPED` callouts so *"a future lint can enforce it"*.

### 9.4 Verification commands that actually decide things

| Claim to check | Command (read-only) |
|---|---|
| "module X landed" | `git ls-files <path>` (empty ⇒ untracked ⇒ at risk, §5) |
| "N tests green" | `git grep -c "def test_" -- <test files>` for the count; re-run scoped pytest only if the tree is quiet |
| "ruff/mypy clean" | `uv run ruff check <allowlist>`; `uv run mypy <module>` — never trust the report on a dirty tree (§9.1) |
| "committed" | `git log -1 --format="%h %ad %an" -- <path>` |
| "the other lane owns H46" | `git grep -n "H46_HYPOTHESIS_ID" -- src` |
| "this file is unchanged vs HEAD" | `git hash-object <f>` vs `git rev-parse HEAD:<f>` (SOTA/21 §6.2's method; do not trust `git diff` mid-churn) |
| "no conflict markers left" | `grep -rn '^\(<<<<<<<\|>>>>>>>\)' <paths>` — measured this lane: **0** in `src/**/*.py`; **27** marker lines in `scripts/finalize_sota.ps1`; **3** each in `.dsh-24x7/HANDOFF.md` and `.dsh-24x7/PROGRESS.md` |
| "duplicate implementation?" | `git grep -n "class <ClassName>" -- src` (§3.1's QRF check) |

---

## 10. Doubled paths and mirror trees

### 10.1 The hazard

A sync artifact creates `src/src/`, `tests/tests/`, `scripts/scripts/`,
`configs/configs/` — full copies of the real trees, one level deeper. Measured
this lane, **OBSERVED**:

| Path | Files on disk | Tracked | Ignored |
|---|---|---|---|
| `src/src` | 426 | 0 | yes (`.gitignore:7`) |
| `tests/tests` | 1,571 | 0 (2 in `DU` conflict state) | yes (`.gitignore:8`) |
| `scripts/scripts` | 12 | **8** (`git ls-files "scripts/scripts/*"`) | no |
| `configs/configs` | 5 | **5** | no |

Root cause is external to Python and was established forensically in
`docs/SOTA/21-concurrent-loop-profile.md` §5: a Mac-side
`dsh-dipcatcher-syncd.sh` loop on a 15-second cadence issues
`sftp put -r src /D:/dipcatcher/src`, and because the remote operand is an
*existing directory*, OpenSSH places the local directory **inside** it. The
same batch's `-put pyproject.toml` / `-put Makefile` / `-put README.md` lines
overwrite those three files with stale Mac copies every cycle. §5.3 proves it by
perfect correlation (only the four `put` dirs doubled; `docs`, which is
pull-only, did not) and §8c.3 confirms it independently (the `C:` clone, same
repo code, has **no** doubled trees). §5.6 shows it recurred at least four
times in git history (`811dffc2` and `621023e8` committed the mirrors;
`7e3ca05` and `bd40510` removed them).

### 10.2 Why agents will not notice

- **Ruff skips them.** `docs/SOTA/19-code-quality.md` §1.1: *"ruff **skips**
  `src/src/` entirely (confirmed: `ruff check src/src` → 'No Python files found
  under the given path(s)', while `--no-respect-gitignore` → `1 I001`). So the
  stale 151-file tree is invisible to lint yet still shipped inside `src/`."*
  Now that `.gitignore:7-8` lists them, every gitignore-respecting tool skips
  them by design.
- **Imports resolve to the real tree.** `.dsh-24x7/PROGRESS.md:73`: *"it is
  preserved, not deleted, and is a provenance/maintenance risk because imports
  resolve to `src/quant_fund`."* Nothing breaks, so nothing gets investigated.
- **They double-count audits.** The same line, plus SOTA/19's scope note
  (670 modules / 152,578 lines) vs my measured 733 tracked `src/*.py` files —
  mirror content silently changes any repo-wide statistic.

### 10.3 The concrete failure: real work stranded outside version control

**OBSERVED, and the clearest instance of the hazard:**

```
scripts/scripts/verify_external_receipt.py:9
    from quant_fund.research.external_receipt import verify_receipt
```

`git ls-files "*external_receipt*"` returns **exactly one path** —
`scripts/scripts/verify_external_receipt.py`. The module it imports exists only
at `src/src/quant_fund/research/external_receipt.py`, which is
**gitignored** (`.gitignore:7`). On a fresh clone the tracked script raises
`ModuleNotFoundError`. Real implementation work is sitting in a directory that
version control has been told to ignore, and a *tracked* file depends on it.

Second instance, same shape: `src/quant_fund/backtest/_fast_kernel.py` is
**untracked** (`?? src/quant_fund/backtest/_fast_kernel.py`), is a 600-line
numba kernel whose bit-identity contract is load-bearing
(`docs/SOTA/13-performance.md:176`, `.dsh-24x7/PROOF.md:48`), and is imported by
two loose root-level scripts (`probe_kernel.py:10`, `run_fallback_tests.py:5`).
`.dsh-24x7/lane-perf/REPORT.md` explains why: the kernel was supposed to be
*inlined* into `fast_replay.py` because a separate file "exceeded the sanctioned
file set" — and `fast_replay.py:57-66` does carry the inlined numba shim, while
the standalone file was never deleted. The allowlist rule was obeyed; the
cleanup was not. **OBSERVED.**

### 10.4 Rules

1. **A path that appears twice is an incident, not a curiosity.** Check
   `git ls-files <path>` immediately: tracked mirror ⇒ needs `git rm` (the
   `.gitignore` comment at `:5-6` says exactly this for `scripts/scripts` and
   `configs/configs`); untracked mirror ⇒ find the writer before deleting
   anything (SOTA/21 §8: *"Deleting the mirrors does not help: the daemon
   recreates them within 15 seconds."*).
2. **Never import from, or write to, a doubled path.** If a module only exists
   under `src/src/`, it does not exist: promote it to `src/` in a lane of its
   own, with a test, and commit it.
3. **Pin pytest `testpaths` explicitly, never bare `tests`.**
   `pyproject.toml:317` lists 5 dirs; ADR-0003 and AGENTS.md both require the
   `tests/tests` mirror stay out of default collection. The 454-duplicate-module
   failure (§7) is what happens otherwise.
4. **Add a structural ratchet.** `docs/SOTA/19-code-quality.md` backlog item 3
   proposes exactly this: *"Add a `tests/unit/test_quality_ratchet.py` assertion
   that no `X/X/` self-nested directory exists under `src`, `tests`, `configs`,
   `scripts`."* **Not implemented as of this lane** — `tests/unit/test_quality_ratchet.py`
   exists but the nested-dir assertion should be verified before relying on it.
5. **Stop the writer before cleaning.** SOTA/21 §8's ordering: pause the sync
   (Mac-side rename of `$ROOT/src`, §7.2.2 — the only kill-free pause), then
   delete, then re-run gates. There is **no documented pause mechanism**
   (§7.1: *"no sentinel file, no lock file, no stop flag, and no documented
   pause procedure anywhere"*).

---

## 11. Worked case study: five concurrent agents on a fragile tree (2026-09-29)

This lane ran inside the incident. Facts as measured, with the orchestrator's
session record where my own measurement window differs.

**Starting state (orchestrator's record, before this lane began):** tree
`ahead 15 / behind 876` vs `origin/main`, 859 files differing, ~30,570
insertions upstream; 138 dirty entries (103 untracked, 35 modified), several
staged; and an **unresolved `git stash pop`** with `<<<<<<< Updated upstream` /
`>>>>>>> Stashed changes` markers in three tracked files —
`src/quant_fund/backtest/engine.py` (2 conflicts),
`src/quant_fund/backtest/fast_replay.py` (17),
`src/quant_fund/cli/main.py` (1 conflict spanning 1,624 lines).
`quant_fund.backtest.engine` did not import at all: `SyntaxError`.

**What I measured (2026-09-29, this lane):** `behind 922 / ahead 24`, later
`behind 934 / ahead 27` (`git rev-list --left-right --count origin/main...HEAD`);
134 → 136 → 156 status entries; `git diff --shortstat HEAD origin/main` =
**907 files changed, 158,667 insertions(+), 25,153 deletions(-)**;
three-dot `git diff --shortstat origin/main...HEAD` (from merge-base
`3dafeb7a`) = **68 files, 21,512 insertions(+)** — the two-dot and three-dot
numbers answer different questions and the brief's figures match neither, which
is itself the §5.3 lesson. The three Python files are now **clean**
(`git status --porcelain -- engine.py fast_replay.py cli/main.py` → empty),
contain **no** markers (repo-wide `*.py` grep → no matches), and sit at
`engine.py` 839 lines, `fast_replay.py` 1,240 lines, `cli/main.py` 187 lines
with `cli/support.py`, `cli/_app.py`, `cli/audit_cmds.py` all tracked — i.e. the
refactored HEAD side won, consistent with the orchestrator's finding that all 38
top-level defs on the stashed side were present in the extracted submodules.
**What is still unresolved right now:** `UU .dsh-24x7/HANDOFF.md`,
`UU .dsh-24x7/PROGRESS.md`, `UU scripts/finalize_sota.ps1` (27 marker lines),
and `DU` on two files inside the gitignored `tests/tests` mirror. Plus
`stash@{0}: On main: pre-sync-20260928` still present. **OBSERVED.**

**Why the resolution needed judgment, not `--ours`/`--theirs`** (orchestrator's
record; consistent with what I can still verify):

- `cli/main.py`: HEAD was a 187-line refactor importing from `cli/support.py`,
  `cli/_app.py`, `cli/audit_cmds.py`; the stashed side was a 1,431-line
  pre-refactor monolith. Verified all 38 top-level defs on the stashed side were
  present in the extracted submodules ⇒ HEAD genuinely superseded it ⇒ take
  HEAD. My measurement confirms the end state (187 lines + three tracked
  submodules).
- `fast_replay.py`: divergence was **two-sided**. HEAD had turnover-cost
  tracking and `exceeds_limit` guards (10 references) that the stash lacked; the
  stash had a `_fast_kernel` extraction (2 references) that HEAD lacked. Neither
  side was a superset ⇒ a real merge was required. My measurement confirms both
  halves of that description are still visible: `fast_replay.py:47` imports
  `exceeds_limit` and `:1252-1257` uses it 6 times (plus `:315` the numba twin
  `_exceeds_nb`), and `_fast_kernel.py` exists but is **untracked** with two
  importers (§10.3).

**The mitigation that made five agents safe** (all three rules are §3):

| Mitigation | Rule |
|---|---|
| Disjoint file allowlist per agent | §3.1 |
| Forbid all 12 state-mutating git commands | §3.2 |
| Forbid repo-wide formatters (`ruff format src tests` would rewrite 1,717 files, including other lanes' in-flight work) | §3.3 |
| Single orchestrator commits serially afterwards | §3.2 |
| Back up conflicted files to a scratch dir before touching them | §5.2 |
| Leave the stash intact (a conflicted `pop` does not drop it) | §3.2 corollary — verified: `stash@{0}` still present |

**OPINION — the transferable lesson.** The conflict was not caused by five
agents. It was caused by *one* earlier `git stash pop` that nobody finished, in
a tree that was 900 commits behind its remote, with 100+ untracked files and an
external daemon rewriting config on a 15-second clock. Concurrent agents
amplify an unresolved tree state; they do not create it. The precondition for
running a swarm is a **quiescent, committed, marker-free tree** — and the
cheapest check is
`grep -rn '^\(<<<<<<<\|>>>>>>>\)' .` plus `git stash list` before any agent
starts.

---

## 12. Operational checklist

**Before launching any concurrent agent:**

- [ ] `git stash list` empty, or every stash accounted for in a tracking file.
- [ ] `grep -rn '^\(<<<<<<<\|=======$\|>>>>>>>\)' .` → no matches in tracked files.
- [ ] `git status --porcelain` reviewed; every untracked path is either
      intentional or committed. Anything untracked is **at risk** (§5).
- [ ] `git rev-list --left-right --count origin/main...HEAD` understood — know
      which side is behind and by how much.
- [ ] External writers identified and paused if possible
      (`docs/SOTA/21-concurrent-loop-profile.md` §5, §7, §8).
- [ ] File allowlists written down and pairwise disjoint (§3.1).
- [ ] Concept/namespace claims written down and disjoint (§6).
- [ ] Each lane has an evidence dir under `.dsh-24x7/lane-<name>/`.

**Per lane, while running:**

- [ ] No state-mutating git (§3.2). No repo-wide formatter/fixer (§3.3).
      No `uv sync` (one shared venv, §7).
- [ ] Scoped gates only (§7). Record the command and its output in the lane dir.
- [ ] Append to `INFLIGHT` with `owner:` / `slice:` / `status:` / `detail:` /
      `note:` — including what was **not** done and why.
- [ ] Self-corrections written into the code and `docs/MATH_SPEC.md`, not only
      into `INFLIGHT` (§9.3).
- [ ] Honesty labels on every number: proper score name, SYNTHETIC/REAL, what
      it does not establish (§8).

**Orchestrator, after lanes finish:**

- [ ] Commit promptly and serially. Close the untracked window (§5.2).
- [ ] Verify claims, not reports (§9): `git ls-files`, `git log -1 -- <path>`,
      scoped re-run, `git grep` for the corrected form.
- [ ] Full gates on a quiet tree: `make lint`, `make typecheck`, `make test`,
      `make fx1-test`, `make fx1-gate`; CI parity via
      `uv sync --frozen --all-groups --all-extras`.
- [ ] Sweep for duplicates introduced by parallel lanes:
      `git grep -n "class <Name>" -- src` (§3.1).
- [ ] Sweep for stranded work: `git status --porcelain | rg '^\?\?'`, and check
      that nothing tracked imports from an ignored path (§10.3).

---

## 13. What this method does NOT establish

Stated plainly, because the rest of this document could otherwise be read as a
productivity claim.

| Not established | Why |
|---|---|
| **That any strategy works.** | The repo's one real-data study is archived with `"verdict": "deflated"` (`research/reality/studies/reality-us-liquid-daily-2026-09-27/receipt.json`; `n_trials: 29`, `dsr: 0.684`, `pbo: 0.122`, `psr: 0.801`). 15 days of output says nothing about markets. `docs/INSTITUTIONAL_READINESS.md`: live readiness is *"Blocked by missing external evidence"*; vendor market data is *"Adapter interface implemented; prospective feed unavailable"*; live broker connectivity *"Not implemented"*. |
| **That agent-authored volume implies correctness.** | 249 agent commits and ~311k tracked Python lines say nothing about defect density. `docs/SOTA/19-code-quality.md` audits that same tree and finds a **P0** fail-open honesty gate, 5 import cycles, 74 cross-module private imports, 4 exception taxonomies needing multiple-inheritance adapters, 6/670 modules with any logging, and a 1,877-line god module whose inlined DCC likelihood **drops an `isfinite` guard** its sibling estimators keep. `docs/SOTA/11-testing-strategy.md` finds **4 of 58** `metrics/` modules have any generated-input coverage — including zero of the scoring modules the honesty contract names first. Volume and rigor are independent axes here. |
| **That the method generalizes.** | One repo, 15 days, one human owner, one product family, Windows + a Mac sync daemon, and an unusually strong pre-existing gate stack (CI matrix, mutation-score records, TLC + Z3 formal lanes, sealed receipts). Removing any of those could remove the result. |
| **That concurrency was net-positive.** | Not measured. There is no controlled comparison against serial work in this repo. The observable facts are that collisions happened and were survivable (§6, §11) and that duplication happened and was **not** noticed (§3.1). **OPINION:** the honest statement is "concurrency was survivable under these controls", not "concurrency was faster". |
| **That the controls are enforced.** | §3.1–§3.3, §4, §12 are prose conventions plus one pre-commit hook (`check-merge-conflict`). Nothing in CI rejects a lane that ran a repo-wide formatter or committed from inside an agent session. `docs/SOTA/21-concurrent-loop-profile.md` §7.1: *"There is no sentinel file, no lock file, no stop flag, and no documented pause procedure anywhere."* The controls hold because the actors followed them. |
| **That the tracked tree is self-consistent.** | §10.3: a tracked script imports a module that exists only at a gitignored path. §3.1: two tracked implementations of the same estimator, one of them unwired. §9.3: a flagged threshold weakness still shipping at its flagged value. These are current, verifiable defects in the tree this playbook describes. |

---

## 14. Changes recommended but deliberately NOT made in this lane

This lane was read-only apart from this file. Each item below is a
recommendation, with the evidence, left for the owner or a scoped lane.

| # | Recommendation | Evidence | Why not done here |
|---|---|---|---|
| 1 | Resolve the three live `UU` conflicts (`scripts/finalize_sota.ps1` 27 marker lines; `.dsh-24x7/HANDOFF.md`, `.dsh-24x7/PROGRESS.md` 3 each) and the two `DU` paths under `tests/tests/`. | `git status --porcelain`, measured 2026-09-29 | Requires state-mutating git (§3.2) |
| 2 | Add `merge=union` for `INFLIGHT` and `day_grind_progress.md` in `.gitattributes`. Commit `52df3277` (Devin AI, 2026-09-28) documents exactly this intent — *"ULTRAPLAN_FRONTIER and day_grind_progress are append-only lab ledgers — the same merge=union treatment RESEARCH_REFERENCES/MATH_SPEC already use, so concurrent PR annotations stop producing textual conflicts"* — but local `HEAD`'s `.gitattributes` carries union merge only for `RESEARCH_REFERENCES.md` and `MATH_SPEC.md` (`git show HEAD:.gitattributes`). The two files that would benefit most are the two append-only ledgers, and `.dsh-24x7/HANDOFF.md` / `PROGRESS.md` are in `UU` state right now — the same conflict class. | `.gitattributes`; `git show 52df3277`; `git status` | Edits a tracked root file; needs a commit |
| 3 | `git rm` the tracked mirrors `scripts/scripts/*` (8 files) and `configs/configs/*` (5), after promoting anything real. `configs/configs/base.yaml` is a **drifted** copy, not a mirror: it lacks the `robinhood_plus:` block that `configs/base.yaml:83-98` carries, so anyone resolving the nested path gets silently different defaults. | `docs/SOTA/19-code-quality.md` §P0-1; `.gitignore:5-6` says these "must be removed with `git rm`, not by ignoring them" | Destructive + needs a commit |
| 4 | Promote `src/src/quant_fund/research/external_receipt.py` to `src/quant_fund/research/`, with a test, so the tracked `scripts/scripts/verify_external_receipt.py` (or its promoted replacement) imports a module that exists on a fresh clone. | §10.3 | Source edit in another lane's territory |
| 5 | Decide `_fast_kernel.py`: commit it or delete it. It is untracked, 634 lines, carries a bit-identity contract, is imported by two loose root scripts, and `lane-perf/REPORT.md` says the standalone file was supposed to be inlined and was. | §5.1, §10.3 | Untracked-file decision belongs to the owner |
| 6 | De-duplicate the wave-9 concept collision: pick one QRF (`models/qrf.py`, which `research/benches_w810.py:24` actually imports, vs `models/quantile_forest.py`) and one WATCH surface (`metrics/watch.py` + `metrics/conformal_martingale.py` vs `models/watch.py`), migrate the tests, delete the loser. | §3.1 | Cross-lane refactor; needs its own allowlist |
| 7 | Apply the flagged `ece_threshold` change (0.05 → ~0.02) in `src/fx1/eval/calibration_eval.py:360`, or record in `INFLIGHT`/`docs/FX1_TRAINING.md` why 0.05 is being kept. | `INFLIGHT:194-196`; §9.3 | fx1 lane edit |
| 8 | Sweep the stale `H46_ranking_data_snooping` prose in `research/catalog/hypotheses.py:270,296,308,313` to `H99`. | §6 step 4 | Catalog edit; the H-id contract is guarded by tests |
| 9 | Add the no-self-nested-directory ratchet to `tests/unit/test_quality_ratchet.py` (proposed in `docs/SOTA/19-code-quality.md` backlog item 3), and a duplicate-class-name check for §3.1's failure mode. | §10.4 rule 4, §3.1 | Test edit; needs a scoped lane + a gate run |
| 10 | Clean the repo root: `_fix.log`, `_probe_fix.py`, `_prof2.txt`, `_t1.log`, `_xbeta_run.ps1`, `probe_*.py`, `prof_fast.py`, `run_*.ps1`, `pytest_full.log`, `.pytest_full.log`, `qlib_run.log`, `northset*.prof`, `mlflow.db` (1.6 MB, gitignored per AGENTS.md but present). Loose root files are how §10.3's stray importers survive unnoticed. | `Get-ChildItem -File` at repo root, measured this lane | Deletions belong to the owner |
| 11 | Link this file from `AGENTS.md` §"Key docs" and from `mkdocs.yml` nav. `docs/SOTA/` is **not** in the mkdocs nav (no `SOTA/` match in `mkdocs.yml`), and `validation.nav.omitted_files: warn` means omission only warns; `make docs` runs `--strict`, so a nav change needs checking. | `mkdocs.yml:22-31`; `make docs` (`Makefile:92-93`) | Edits tracked root files |

---

## 15. Evidence index

Every non-obvious claim above, with the command or location that supports it.
All measurements taken 2026-09-29 in `D:\dipcatcher` during a read-only
documentation lane; `HEAD` moved `57d305dd → abc7a7d0` while they were taken.

```bash
# authorship / volume
git shortlog -sn HEAD                                  # §1 table
git rev-list --count HEAD                              # 597 at 57d305dd
git log --reverse --format=%ad --date=short | head -1   # 2026-09-15
git log -1 --format=%ad --date=short                    # 2026-09-29
git ls-files "*.py" | Measure-Object                    # 1,871 files / 311,103 lines

# divergence (moving target — §0, §5.3, §11)
git rev-list --left-right --count origin/main...HEAD    # 922 24 → 934 27
git diff --shortstat HEAD origin/main                   # 907 files, +158667 / -25153
git diff --shortstat origin/main...HEAD                 # 68 files, +21512 (from merge-base 3dafeb7a)
git status --porcelain | Measure-Object                 # 134 → 136 → 156
git stash list                                          # stash@{0}: On main: pre-sync-20260928

# unresolved conflicts (§11, §14.1)
git status --porcelain | rg '^(UU|DU|UD|AA|DD|AU|UA)'
rg -n '^(<<<<<<< |=======$|>>>>>>> )' scripts/finalize_sota.ps1 .dsh-24x7/*.md
rg -n '^(<<<<<<< |>>>>>>> )' -g '*.py'                  # no matches

# purge incident (§5)
git branch -a | rg purge                                # remotes/origin/chore/purge-derived-history
git log --oneline origin/chore/purge-derived-history    # tip 6ebb312d; 5c988bfc; a6f178d5
git status --porcelain -- src/quant_fund/backtest/_fast_kernel.py   # ?? (untracked)

# H-id protocol (§6)
git grep -n "H4[6-9]_HYPOTHESIS_ID\|H5[01]_HYPOTHESIS_ID\|H99_HYPOTHESIS_ID" -- src
#   src/quant_fund/research/catalog/hypotheses.py:274-275, 991-996, 1319-1334
git grep -n "H46_ranking_data_snooping" -- src           # 4 stale refs: :270, :296, :308, :313

# duplicate wave-9 implementations (§3.1)
git ls-files src/quant_fund/models/{qrf,quantile_forest,watch,enbpi}.py \
             src/quant_fund/metrics/{watch,conformal_martingale}.py
git log -1 --format="%h %ad %an" -- src/quant_fund/models/qrf.py              # ffaf1f76 devin bot
git log -1 --format="%h %ad %an" -- src/quant_fund/models/quantile_forest.py  # 6c47b032 Advaith
git grep -n "class QuantileRegressionForest" -- src      # qrf.py:79, quantile_forest.py:90
git grep -n "models.qrf\|models.quantile_forest" -- src   # only benches_w810.py:24 uses qrf

# mirror trees (§10)
git ls-files "scripts/scripts/*"                         # 8 tracked
git ls-files "configs/configs/*"                         # 5 tracked
git ls-files "*external_receipt*"                        # only the tracked script
Get-ChildItem -Recurse -File -Filter external_receipt.py # only src/src/...
git show HEAD:.gitignore | Select-Object -First 19       # /src/src/ /tests/tests/ …
git log -1 --format="%h %ad %an %s" -- .gitignore        # fd22fbf4 cursor-agent[bot]

# honesty contract (§8)
git grep -n "FORBIDDEN_RESEARCH_METRIC_KEYS = " -- src   # research/catalog/registry.py:101
git grep -n "FORBIDDEN_HEADLINE_TOKENS" -- src/fx1       # honesty.py:42
rg '"verdict"' research/reality/studies/*/receipt.json   # "deflated"

# gates (§7)
# Makefile:10-40 (sync/test/test-full/lint/fmt/typecheck), :105-128 (fx1-*), :187 (receipts-reverify)
# .github/workflows/ci.yml:52, 198, 385-441, 486-497, 737-740
# .github/workflows/fx1.yml:36-63 (honesty inheritance is a blocking step)
# pyproject.toml:317 (testpaths), docs/adr/0003-fx1-test-lane-separation.md
```

Tracked artifacts quoted: `AGENTS.md`; `INFLIGHT` (lines 1-4, 16-17, 23-26,
64-72, 77, 107-110, 120, 136, 145-156, 164-166, 170-173, 180-181, 190-198,
204-209, 218-225); `day_grind_progress.md`; `docs/DATA_CONTRACTS.md`;
`docs/SOTA_CANON_ROADMAP_2026_09.md` §4; `docs/SOTA/11-testing-strategy.md`
§4.1, §4.6, header; `docs/SOTA/19-code-quality.md` §0, §1.1-1.3, P0-1, P0-2,
P1-1…P3-3, backlog 1-4; `docs/SOTA/21-concurrent-loop-profile.md` §1, §3.1,
§5.1-5.7, §6.1-6.3, §7.1-7.2, §8, §8b-8d, §9; `docs/adr/0001`, `0003`, `0008`;
`docs/INSTITUTIONAL_READINESS.md`; `docs/FX1_TRAINING.md` (marker convention);
`docs/OPERATIONS_RUNBOOK.md`; `docs/SOTA/13-performance.md:176`;
`.dsh-24x7/HANDOFF.md` (L21, L106, L112, L127-129, L145-147, L190-191, L200,
L209); `.dsh-24x7/PROGRESS.md` (L72-73, L529-530, L551-552, L676-678);
`.dsh-24x7/lane-perf/REPORT.md`; `.dsh-24x7/PROOF.md:48`;
`.dsh-24x7/pre-restore-status.txt`; `.dsh-24x7/g-allow{,2,3}.txt`;
`.pre-commit-config.yaml`; `.gitattributes`; `mkdocs.yml:13-31`.

Orchestrator-supplied session facts used in §11 are marked as such where my own
measurement window could not reproduce them.
