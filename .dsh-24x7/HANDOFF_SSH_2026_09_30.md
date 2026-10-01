# Handoff note — local macOS → SSH (2026-09-30 ~16:20 IST)

Written for a session continuing on the remote host. Everything below was
**observed**, not assumed. Note up front: the local checkout changed
substantially *while this note was being written* (a concurrent session
committed, rebased and pushed mid-measurement), so §1 records a re-measured
state as of 16:19 IST. Re-verify before acting — timestamps are load-bearing.

- Local root: `/Users/vaithianathan/dipcatcher`, branch `main`
- Remote root: `D:\dipcatcher` on `winpc` / `ah-remote` / `codex-remote`
  (all four ssh aliases in `~/.ssh/config` resolve to the same box,
  `DESKTOP-AJN4V4Q` @ `100.116.120.51`, direct tailscale)

---

## 0. Read this first: a concurrent session is LIVE on the local checkout

Do not assume anything you read here is still true 5 minutes from now.
Observed churn during this handoff:

| time | observation |
|---|---|
| 15:53 | `git status`: 15 modified + 2 untracked; **2 ahead / 2 behind** origin |
| 16:00 | `test_fourier_pricing.py`: **5 test failures**, 7 ruff `F401` |
| 16:07 | concurrent session commits `2a14fa470` + `73df5cfdb` |
| 16:08 | same fourier file **green** (all tests pass) |
| 16:14 | history **rewritten** — rebased onto `c5ff85456`, new shas, **pushed**; divergence gone |
| 16:17 | ruff: 6 errors → 3 errors → 1 error (converging live) |
| 16:22 | `ruff check src tests` → **All checks passed**; fourier file now **+2,207 / −7** |
| 16:26 | `ruff format --check` → **2,153 files already formatted** (fully green) |

The 5 fourier failures I caught at 16:00 were **mid-edit and already fixed by
the other session**. Do not "repair" `test_fourier_pricing.py` on the assumption
it is broken — re-measure first.

**As of 16:26 both ruff gates are fully green.** The only thing left in
`git status` is `M tests/unit/models/test_fourier_pricing.py` (uncommitted but
clean per lint/format) plus this note. That file grew +1,732 → +2,207 in three
minutes during the churn above; its last edit was **16:23** and it was still
quiet at a 16:27 recheck, so the other session has likely stopped editing —
but it has **not committed** the expansion. Re-measure before touching it.

---

## 1. Git state — reconciled locally, but the REMOTE IS 5 COMMITS BEHIND

**Local (as of 16:19):**

```
HEAD == origin/main == e00ab310c     0 ahead / 0 behind
git status:  M tests/unit/models/test_fourier_pricing.py
             ?? .dsh-24x7/HANDOFF_SSH_2026_09_30.md   (this file)
```

Clean. The divergence I saw at 15:53 is **gone** — the concurrent session
rebased the local wave-16/17 work onto the PR #441 merge and pushed it.
History was rewritten, so **old shas from before 16:14 are invalid**
(`1e8a295fc` and `6fe0b37d9` no longer exist; they are now `51d81a872` and
`55a701fee`).

**The 5 commits the remote does NOT have** (`git log --oneline c5ff8545..HEAD`):

```
e00ab310c style: apply ruff format to fx1/eval facade + fleet_eval
73df5cfdb feat(models): wave 17 — DiffPTS full-ELBO diffusion probabilistic forecaster
2a14fa470 fix(arch): land layering/boundary repairs — 4 guard reds root-caused, 0 violations
51d81a872 chore(quality): register adaptive_eps type-ignores in manifest
55a701fee feat(research): wire wave 16 as 6 OPTIONAL scorecard families (74 -> 80)
```

**Remote (verified over ssh):**

```
HEAD              c5ff8545   (the PR #441 merge — 5 behind origin/main)
vs origin/main    5 ahead?  -> NO: "5  0" = 5 behind, 0 ahead
                    (left-right count origin/main...HEAD)
status            ?? tests/tests/     (1,131 files — the dropped mirror tree)
branch            `git branch --show-current` returns EMPTY
                    -> remote is in DETACHED HEAD at c5ff8545
python            .venv\Scripts\python.exe -> Python 3.12.10
disk D:           ~3.73 TB free of ~5.4 TB
fleet heartbeat   .dsh-24x7\fleet_heartbeat.json  ABSENT -> no fleet running
python processes  none
```

⚠️⚠️ **STOP — the remote is NOT simply "behind". Do not hard-reset it.**

`git branch -vv` on the remote:

```
* (HEAD detached at c5ff8545)   c5ff8545  Merge pull request #441 ...
  main   4005afc7 [origin/main: ahead 28, behind 984]  checkpoint before checking out main
```

Two separate facts, and the second one is dangerous:

1. **HEAD is detached** at `c5ff8545` — that is the PR #441 merge, 5 behind
   `origin/main`. Harmless on its own.
2. **The remote's local `main` branch is a stale checkpoint at `4005afc7` that
   is 28 ahead / 984 behind origin.** Those 28 commits are *not* the same 5
   commits local just pushed.

**I verified those 28 commits are NOT already on origin** — so they are
genuinely unlanded work, and `git reset --hard origin/main` would destroy them.
Checked three ways, all negative:

- `git log origin/main --grep='<subject>'` for 5 of the subjects → **0 matches
  each**
- `git grep merkle_root_hex_v2 origin/main -- src` → **not found**
- `git grep 'wis_decomposition\|fair_crps' origin/main -- src/quant_fund/metrics`
  → **not found**
- Same greps against the local working tree (which *is* at origin/main) →
  **also not found**. `threshold_energy_score` does exist
  (`metrics/energy_score.py`), but the fairness/WIS/malleability work does not.

Sample of the 28 (`git log --oneline origin/main..main` on the remote):

```
4005afc7 checkpoint before checking out main
abc7a7d0 fix(audit): write the ledger in binary mode so it survives Windows
92801b4e fix(pit): fsync the write descriptor, not a read-only reopen
7280ea7f fix(fx1): resolve agent-gw config home via HOME, not USERPROFILE
4b02af02 fix(fx1): harden honesty matcher — unicode fold, boundaries, claim proximity
7dd19651 feat(validation): assert CPCV purge/embargo at bar level, add worst-path ranking
ff1446b4 feat(metrics): add opt-in multiplicity control for pairwise DM comparisons
6d8dfc57 feat(metrics): add fair CRPS, proper threshold-weighted CRPS, WIS decomposition, skill scores
ffa24554 fix(metrics): correct threshold_energy_score propriety claim, warn at weight >= sqrt(2)
0c40247e fix(proofcore): add malleability-resistant merkle_root_hex_v2; deprecate v1
9f4f55ad ci: add composite uv-setup action and harden workflow hygiene
fd22fbf4 chore: ignore doubled-path mirror trees that break repo-wide tooling
...
```

Note these are mostly **Windows-portability and honesty/metrics-correctness
fixes** — exactly the class of thing that is easy to lose and hard to
rediscover. `92801b4e` (fsync the write descriptor, not a read-only reopen) is
one of the two `paper/ledger.py` Windows bugs recorded in `.dsh-24x7/HANDOFF.md`
and `INFLIGHT`; `abc7a7d0` is the same family.

**How stale is it?** Merge base of remote `main` and `origin/main` is
`3dafeb7a8` — `fix(ci): archive decided reality study; document pending/decided
ledger lifecycle (#201)`, dated **2026-09-28 08:26**. So the remote branched
off only ~2 days ago, but origin has moved **984 commits** since (that is the
waves 12–17 push). The rebase is therefore 28 commits onto 2 days of very dense
upstream churn — non-trivial, but the 28 are small and mostly surgical, and
they touch areas (Windows fsync, honesty matcher, metrics propriety, proofcore
merkle) that the waves largely did *not*, so conflicts should be localized.

**Correct first move on SSH — preserve, then reconcile:**

```powershell
cd D:\dipcatcher
git branch backup/remote-main-4005afc7 main      # 1. pin the 28 commits to a name
git log --oneline origin/main..backup/remote-main-4005afc7 > ..\remote_only_28.txt
                                                   # 2. record the list off-tree
git checkout main
git rebase origin/main                             # 3. replay the 28 onto current origin
                                                   #    (984 behind -> expect conflicts;
                                                   #     do NOT use -X theirs blindly)
```

Only if `git rebase` shows the 28 are *entirely* superseded upstream (rebase
empties out) is a reset safe. Assume they are not until proven.

Then confirm the two wave-17 files arrived — they were **absent** on the remote
at 16:18:

| Artifact | local | remote (16:18) |
|---|---|---|
| `src/quant_fund/research/benches_w16.py` | present | **MISSING** |
| `src/quant_fund/models/diffusion_forecaster.py` | present | **MISSING** |
| `tests/unit/models/test_diffusion_forecaster.py` | present | **MISSING** |
| `src/quant_fund/models/greek_neutral_portfolios.py` | present | present |

**`pyproject.toml` changed on origin** via `28a6d7fa8` (now in local history):
adds `license = { file = "LICENSE" }`, two classifiers (incl.
`Private :: Do Not Upload` — a deliberate second guard against accidental PyPI
upload), and `LICENSE` to hatch's `only-include`. Metadata-only, so it should
not invalidate resolution, but per `AGENTS.md`:

```bash
make sync          # uv sync --frozen --all-groups --all-extras  (must pass)
```

If `--frozen` errors on the remote, run `uv lock` and commit the regenerated
lockfile — do **not** drop `--frozen`; CI parity (`.github/workflows/ci.yml` +
`fx1.yml`) depends on it.

---

## 2. What the 5 pushed commits actually contain

Nothing is uncommitted any more except the fourier file. Summary for review:

**`55a701fee` — wave 16 as 6 OPTIONAL scorecard families (74 → 80).**
`research/benches_w16.py` (377 LOC) float-only adapters over the lane bench
functions (rwcv/xva precedent); `registry.py` +16/−0, `agent.py` +14/−0,
REQUIRED + ORDER untouched. Families: `vol_loss_decomposition` (the flip:
`vld_ratio_raw` 5.01 → `vld_ratio_aligned` 0.052, breach narrowing 0.959),
`hierarchical_conformal` (min coverage 0.91, widths 23.1 → 14.4 over m=0..10),
`multisource_conformal`, `extra_tilt` (29.0% paired length reduction,
ESS 29.9), `forecast_selection` (scale-free basis admits ZERO diluters, 67.5%
dilution removal; equal-weight basis replicates the paper's Table-10 failure;
pair-level mirror R² 0.9978 — key-semantics deviation documented),
`rl_market_maker` (torch-gated tiny-budget; `sim_internal_*` PnL keys excluded).
Battery 6.8 s / 60 s envelope. `verify-research`: `scorecard_families=80`,
`errors=[]`, `claim=research_only`; research suite 2883 passed / 0 failed.

**Honest negative recorded — keep it recorded:** `multisource_conformal` gap
degradation is **FLAGGED** (vacuity 0.22 vs 0.0017, `bound_vacuous_gap` 1.0).

**`2a14fa470` — arch/layering repairs, 4 guard reds root-caused → 0 violations.**
Per-edge adjudication, source fixes preferred over baseline pins:

- *Lazy-import cycle-breakers* (sanctioned by `docs/ARCHITECTURE_GUARDS.md`,
  `41fb78a0` precedent) — import-time upward edges moved inside function
  bodies, single-source-of-truth preserved:
  - `metrics/forecast_selection.py` → new `_equal_weights()` wrapper (metrics
    sits *below* models); 4 call sites updated
  - `microstructure/zi_lob_simulator.py` → `as_optimal_quotes` inside
    `avellaneda_stoikov_quotes`
  - `models/regime_conformal_var.py` → `stress.regimes` inside
    `_synthetic_regime_stream` (only the SYNTHETIC bench DGP needs it)
  - `reporting/regime_performance.py` → `stress.bundle` inside
    `bundled_h15_rate_map`
- *proofcore isolation* — `LAZY_WHITELIST["proofcore"]` tightened from
  `{"utils"}` to `frozenset()`: proofcore is standalone, **no** quant_fund edge,
  lazy included (`configs/arch_boundaries.toml` + leakage rule LH011). So
  `proofcore/cli.py` gained an inlined byte-equivalent crash-safe
  `_atomic_write_text` (mkstemp + fsync + `os.replace` + directory fsync)
  replacing `utils.atomicio` at 3 call sites.
- *Orphaned-repair recovery* — `proofcore/ci.py` re-applies the redsem-branch
  fix `a81f4844` that was **lost to union-merge drift**: restored the missing
  `report` element in the coverage argv (a real bug) + added
  `_receipt_verifier_command()` routing `receipt.v2` / `receipt_sha256`-sealed
  receipts to `verify-receipt`, else `verify-research` (narrow except tuple).
- *Two justified (b) pins* — `fx1-harness-surface allow_only` += exactly
  `quant_fund.models.options` + `models.iv_approx` (oracle-by-design edges the
  options-reasoning eval bank needs; wave commit `57ca9c23` had updated the
  atlas but not the toml — this aligns the lagging guard; explicitly **not**
  `models.**`), and `leakage THIRD_PARTY_WHITELIST` += pandas/polars (the W8 IO
  guard must import the readers it interposes; byte-identical to adjudicated
  `41fb78a0`).
- *Atlas pins* += two `fx1.eval.rubric_banks` → `metrics.conformal` /
  `metrics.scoring` edges; generated artifacts regenerated (they were stale at
  HEAD).

  ⚠️ **`docs/ARCHITECTURE_ATLAS.md`, `docs/architecture/manifest.json`,
  `docs/architecture/module_deps.mmd` are TRACKED generated files.** They now
  reflect the ~30 modules from waves 12–15. If any lane is dropped at review,
  regenerate (`python scripts/gen_arch_diagrams.py`) or the atlas test goes red.

  Gates quoted in the commit: arch+layering+leakage+proofcore+atlas 67 green;
  `check_import_boundaries` 0 violations / 894 modules; `gen_arch --check`
  fresh; regression ring 136 green.

**`73df5cfdb` — wave 17: DiffPTS full-ELBO diffusion forecaster.**
Ye, Li, Liu, Jiang, Sekimoto & Jiang 2026 (NeurIPS, arXiv:2609.32363),
fetched + verified. 1,024 LOC module + 730 LOC test (17 tests, 29.9 s).
LSNM forward process with learned mean/scale (Eq. 9), N(f,g) prior (Eq. 15),
posterior coefficients tested `c0+c1+c2=1` (Eq. 11), exact ELBO (Prop 3.1)
available via `weighting='elbo'` and tested against Eq. 12 at init; the
*trained* objective is Prop 3.3 / Eq. 14 (Appendix E.4 confirms the paper's own
simplified form); Algorithm-2 sampler tested analytically at T=1;
linear-schedule anchor reproduces the paper's quoted `alpha_bar_T ~ 0.82`
(got 0.8168). Torch-gated training, **pure-numpy inference** on extracted
float64 params. Shared-stream proper scores: CRPS 0.5036 / 0.4929 / 0.5055
(gauss / t4 / bimodal), ≤ NGBoost everywhere with ≥4× tolerance margins, dead
heat with TORF on t4. Two documented FINDINGS: exact-ELBO weighting degenerates
under cosine schedule; scale-normalized denoiser inputs weaken `g`
identification. Honest bimodal-PIT limitation (tiny denoiser smooths narrow
clumps) pinned in tests.

**`51d81a872`** — registers `adaptive_eps` type-ignores in the quality manifest
(e-PS commit follow-up).
**`e00ab310c`** — ruff format only (blank-line normalization on the fx1/eval
facade + `fleet_eval` adapters).

**Stashes (2, old, unrelated — do not lose them):**

- `stash@{0}` — WIP on `main` at `c2d360b6`, 12 files / +167 −23
  (`test_torch_backend_edges`, `test_regime_eval`, `test_tearsheet`,
  `test_schema_edges`, …)
- `stash@{1}` — WIP on `main` at `4b7ff90`, 42 files / +771 −101
  (`quant_models/cli.py`, `schemas/market.py`, `test_perp_engine.py`,
  `test_quantile_signals.py`, …)

---

## 3. Gate state on the local machine

| Gate | Result (16:26) |
|---|---|
| `ruff check src tests` | **All checks passed** (was 6 errors at 16:17) |
| `ruff format --check` | **2,153 files already formatted** — clean |
| `mypy src/quant_fund` | Success, **823** source files |
| `mypy src/fx1` | Success, **71** source files |
| `mypy --strict` facade (`__init__.py` + `public.py`) | Success, 2 files |
| `scripts/check_import_boundaries.py` | 0 violations, 894 modules, 10 baselined, **0 stale** |
| `scripts/gen_arch_diagrams.py --check` | fresh |
| `test_diffusion_forecaster.py` | **17 passed** |
| `test_fourier_pricing.py` | green as of 16:08 |

mypy file counts grew vs the `INFLIGHT` waves-12–15 close-out entry
(716 → 823 `quant_fund`, 69 → 71 `fx1`) — waves 16/17 landed since.

**Known pre-existing flake (do not chase):**
`tests/unit/pretrade/test_latency.py::test_allow_path_meets_latency_gate` — a
wall-clock P50/P99 ns budget that fails *only* under xdist parallel load; green
standalone (verified twice across two `make test` runs). Has a
`_coverage_running()` exemption but no parallel-load exemption. Owner options
recorded in `INFLIGHT`: add a load exemption, or move it to the perf lane.

**Local disk pressure — the actual reason to shift to SSH:**
17 GiB free of 228 GiB (92% used — was 18 GiB / 91% at 16:06, so it is still
creeping down). `.git` **5.7 G**, `.venv` 2.2 G, `data`
676 M, `.dsh-24x7` 101 M. A prior session already scoped reclaim and found the
small dirs negligible (`.benchmarks` 0 B, `dist` 3.3 M, `third_party` 25 M,
`.hypothesis` 3.6 M, `git rev-list --objects --all` = 16,309) — any real win
must come from `.git` or `data`. The remote has ~3.73 TB free.

---

## 4. Suggested sequence once you're on SSH

1. **Fix the detached HEAD and catch up** (§1). Confirm `benches_w16.py` and
   `diffusion_forecaster.py` now exist on the remote.
2. **Re-verify the lockfile** — `make sync` (`--frozen`) after the
   `pyproject.toml` metadata change lands. `uv lock` only if `--frozen` fails.
3. **Re-run the full gates on the remote before trusting anything.**
   Linux/Windows has bitten here: `paper/ledger.py` had two Windows-only bugs
   (`os.fsync` on a read-only fd → EBADF; `is_absolute()` missing POSIX
   anchors). Expect the inverse class of surprise now that work moves *to*
   Windows.
4. **Leave `test_fourier_pricing.py` alone** unless the concurrent session has
   stopped — it is still being edited and is the only red thing (§0).
5. **Full gate ladder** per `AGENTS.md`: `make lint`, `make typecheck`,
   `make test` (PR gate), `make fx1-test`, plus `make fx1-gate` since fx1
   changed (`src/fx1/eval/__init__.py` + the two `rubric_banks` arch pins).
6. **Remote `tests/tests/` — 1,131 untracked files.** `AGENTS.md` says the
   mirror "was dropped; if a stray reappears, keep it out of default
   collection." It reappeared. Leave it untracked and confirm `pytest.ini`
   testpaths (`tests/unit`, `tests/property`, `tests/regression`,
   `tests/end_to_end`) still exclude it — an older note records **454
   duplicate-module collection errors** when it got swept into
   `testpaths = [tests]`.

### Remote fleet conventions (load-bearing — jobs break silently otherwise)

- **PowerShell only.** `cmd /c` inner strings need backtick-escaped quotes.
  My `ssh winpc 'cmd /c "cd /d D:\dipcatcher && git ..."'` attempt silently
  returned only the shell banner; `powershell -NoProfile -Command` worked.
  **Use the latter.**
- **WMI spawn, not `Start-Process`** — `Invoke-CimMethod -ClassName
  Win32_Process -MethodName Create`, wrapped in
  `cmd /c "... 1> stdout.log 2> stderr.log"` (no built-in redirection), so jobs
  survive ssh session teardown.
- **Parametrized launcher:** `scripts/fleet_spawn.ps1 -Manifest jobs.json` +
  `scripts/fleet_watchdog.ps1` (heartbeat at
  `.dsh-24x7\fleet_heartbeat.json`, auto-respawn bounded by `-MaxRespawns`);
  `scripts/fleet_manifest_sota.ps1` regenerates the canonical SOTA manifest.
  Prefer over the legacy `spawn_*.ps1` one-offs.
- **Thread pinning** in each job's `env` block: `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`.
- **Durable paths:** artifacts under `.dsh-24x7\`, weights under
  `data\models\`, bars under `data\raw\sources\`. Never temp dirs.
- **Defender exclusion** for `D:\dipcatcher`, or parquet/bar reads get throttled
  mid-fleet.
- Remote `D:\dipcatcher` has diverged from `D:\evalenv`'s editable install
  before (no `data/sources/` subpackage — hedge_lab lineage), which breaks
  `collect_binance_deep.py`. Collect deep bars locally and scp.
- The Microsoft Store python shim suspends spawned children on that host; watch
  for stray `python3.12.exe` resolving to `WindowsApps`.
- `D:\evalenv` has no pip — use the bench venv (`D:\bench-qlib`) or a
  `sys.path` runner for remote tests.

---

## 5. Research-programme context (so the next session can pick a lane)

`INFLIGHT` (819 lines) and `day_grind_progress.md` (2,066 lines) are the
authoritative work trackers — **read them before grabbing work**. Both are
tracked in git. Shape as of now:

- **Scorecard families: 80** (was 36 before waves 12–15; 74 after w15; +6 from
  w16). `verify-research` reported `errors=[]`, `claim=research_only` at every
  step.
- **Waves 12–15 grand total** (from the close-out entry): ~30 new research
  modules, ~2,000 new tests, 6 real bugs caught and fixed (large-deviations
  signs ×2, subspace bootstrap inversion, `capability_value` Cap-invariance +
  fixture starvation, the fourier COS 5-defect chain, `verify.py` p=0.0
  boundary via mutation testing), ~25 citation corrections against fetched
  sources, 2 paper-level findings (cash-OE Table-1 infeasibility +
  symmetric-legs impossibility).
- **Wave 16** = the 6 families above. **Wave 17** = `greek_neutral_portfolios`
  (Tan, Roberts & Zohren 2026, arXiv:2609.33767) + `diffusion_forecaster`
  (DiffPTS).

**Untouched backlog** (`docs/SOTA_WAVE13_BACKLOG.md` Tier D unless noted):

- Tier B4-ii — C51 RL layer on `zi_lob_simulator`
- ExTRA conformal (2609.30886), generalized hierarchical CP (2608.15500),
  MS-RLCP (2609.14531), forecast-selection dilution (2609.26303),
  vol loss-vs-model decomposition (2609.27024), SGA multi-step UQ (2609.28582)
  — *note: w16 already wired `extra_tilt`, `hierarchical_conformal` and
  `forecast_selection` families, so re-check which of these are genuinely open*
- Owner calls: fx1 `capability.py`/CLI wiring for `ext_bench` +
  `options_reasoning` (recipes in the lane reports); full 210-mutant campaign on
  `verify.py` + catalog
- `pretrade/test_latency.py` xdist flake (§3)

**Honesty contract — unchanged and binding.** Proper scores only (pinball,
CRPS, PIT, QLIKE, Brier, ECE, Kupiec, HMM likelihood); never headline
Sharpe/Sortino/Calmar/P&L/NAV. `FORBIDDEN_RESEARCH_METRIC_KEYS` in
`quant_fund.research.catalog` is mirrored by
`fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` — change together;
`tests/fx1/test_honesty_inheritance.py` blocks drift. SYNTHETIC results always
labelled, never market evidence. No live-trading claims (no broker
connectivity; see `docs/INSTITUTIONAL_READINESS.md` for the five
minimum-evidence conditions). Receipts immutable; every claim reproducible from
a receipt hash.

Also relevant now that origin carries a proprietary LICENSE: `28a6d7fa8` added
`Private :: Do Not Upload` to classifiers specifically so an accidental public
PyPI upload fails. Don't undo that when touching `pyproject.toml`.

---

## 6. One-line summary

Local `main` is **clean and in sync with origin at `e00ab310c`, all lint/format
/type/arch gates green** (the concurrent session rebased + pushed waves 16/17
and the arch repairs while this note was being written). The real handoff
problem is on the **remote**: it sits in a **detached HEAD at `c5ff8545`**, and
its local `main` is a stale checkpoint at `4005afc7` that is **28 ahead / 984
behind origin — and I verified those 28 commits are NOT on origin** (fair-CRPS /
WIS decomposition, `merkle_root_hex_v2`, two Windows fsync + binary-mode
portability fixes, honesty-matcher hardening, CPCV purge assertions, composite
uv-setup CI action). So: **branch-backup the remote `main` before touching it
and rebase it onto origin — never `reset --hard`** — then `make sync`
(`--frozen`; origin changed `pyproject.toml` for the proprietary LICENSE) and
re-run the gate ladder, since Linux→Windows portability has bitten here before.
