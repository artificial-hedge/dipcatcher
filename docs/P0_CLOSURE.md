# P0 Closure — Open Evidence Debt (ULTRAPLAN §"P0 — Close open evidence debt")

Status date: 2026-10-07. Owner: `evidence-debt` (task-2). This file is the
authoritative, unflinching P0.1–P0.7 status table. Every row is either
**closeable-now (with evidence path)**, **runbook-ready (with path)**, or
**blocked (with the precise blocker)**. Nothing here is marked closed on the
strength of a green test suite; fleet evidence is claimed only where the
receipt files named below actually exist in `.dsh-24x7/`.

Honesty contract: all rows are research-only proper-score evidence
(pinball/CRPS/PIT/QLIKE); no live-trading or P&L claims. Nothing in this file
was fabricated to fill a cell — where evidence is absent the row says so.

A hash-pinned inventory of every evidence file cited here lives at
`quality/p0_evidence_inventory.json` (regenerate: `uv run python
scripts/build_p0_inventory.py` equivalent inline block documented below), and
the inventory itself is sealed as a `receipt.v2` at
`receipts/p0_evidence_inventory_v1.json` (path reported to the Lead for
`docs/evidence/index.md` regeneration, which is Lead-owned).

## Status table

| # | Item | Verdict | Evidence path / precise blocker |
|---|------|---------|--------------------------------|
| P0.1 | Poll remote fleets (`h4f_*`, `tfmfix_*`, `nd_*`/`nh_*`, `s11_*`); respawn dead shards; capture stderr for shards that die twice | **runbook-ready** (poll+respawn+capture); **blocked** for execution | Runbook §P0.1 below (uses `scripts/fleet_spawn.ps1` + `scripts/fleet_watchdog.ps1` per AGENTS.md fleet conventions). BLOCKER: no SSH execution path from this session to the Windows host `D:\dipcatcher`; the WMI spawn/respawn steps must run there. Local completeness audit (which shard files exist and which evidence JSONs reference them) is reflected in `quality/p0_evidence_inventory.json`. |
| P0.2 | TimesFM contract-v2 splice (`scripts/splice_timesfm_fix.py`) on all v4-daily + h4f cells; challenger/target non-TimesFM columns bit-identical; pre-splice matrices archived | **runbook-ready**; deep cells **blocked** | Splice artifacts already evidenced for the v3 d1/h4 cells: `.dsh-24x7/eval-shards/*.tfmv2.npz` (post-splice) with the pre-splice originals retained alongside; splice receipt `.dsh-24x7/evidence-sota-tfmfix-probe.json`. Runbook §P0.2 runs `--check-only` then the splice for the v4/h4f cells. BLOCKER: the v4-deep and h4f part matrices live under `.dsh-24x7\eval-full\` on `D:\dipcatcher` only (not present in this checkout), so bit-identity assertions for those cells cannot be executed here. |
| P0.3 | Contract-v2 re-merge all cells (`--bars-root data/raw/sources`, `n_boot` 2000, fresh receipts) → `docs/EVAL_REPORT_SOTA.md` §4.1 | **closeable-now (merged receipts cited)**; full independent re-merge **runbook-ready**, remote parts **blocked** | Merged receipts exist and are the §4.1 sources: `.dsh-24x7/merge_d1_v2aug.json`, `.dsh-24x7/merge_h4f_v2aug.json`, `.dsh-24x7/evidence-sota-eval-v4-d1-v2.json`, `.dsh-24x7/evidence-sota-eval-h4f-v2.json`, `.dsh-24x7/evidence-sota-eval-v3-d1-tfmv2.json` (hash-pinned in `quality/p0_evidence_inventory.json`). Their `source_parts_sha256` blocks bind the per-shard inputs. BLOCKER for a from-scratch re-merge: `eval-full` part matrices are remote-only (see P0.2). NOTE: `n_boot`/contract fields inside each receipt are the machine-checked record; the row-level re-audit of `n_boot == 2000` was NOT re-executed in this session — treat as runbook step P0.3.v. |
| P0.4 | Native-protocol crossover merge (`sota_eval_native.py --merge-parts .dsh-24x7/native/nd_*`) → §4.3 ordering table | **closeable-now** | Merge receipts `.dsh-24x7/native/MERGED_d1_native.json` and `.dsh-24x7/native/MERGED_h4_native.json` exist; per-cell parts `native/nd_*` (11 daily, incl. eth/xrp deep) and `native/nh_*` (5 4h deep) + `ns4_*` are present locally and bound by those merges. §4.3 ordering table is derived from these two receipts. |
| P0.5 | `s11_*` + `s23_*` corrected-pairing seed replication → §4.4 / B1 closure | **closeable-now** | Merge receipts `.dsh-24x7/merge_s11_v2aug.json` and `.dsh-24x7/merge_s23_v2aug.json` (plus the earlier `merge_seed23_aug.json`) exist and are the §4.4 sources. |
| P0.6 | Full-coverage 4h merge replacing complete-case receipt; student-t coverage row (hardened fitter 0-NaN over 1500 origins) RECEIPTED, not just probed | **closeable-now (merge)**; claim re-verification **runbook-ready** | Full-coverage merge receipt: `.dsh-24x7/merge_h4fix_aug2.json` (replaces the complete-case-only receipt). Coverage rows: `.dsh-24x7/evidence-sota-eval-h4f-v2.json` (`coverage` block: per-model `emitted`/`finite_crps`/`missing_or_failed`/`per_asset`) — §4.2 cites it for the 0-NaN claim. This session did NOT re-derive the row-level counts from the losses files; runbook §P0.6.v re-checks them against `.losses.npz`. |
| P0.7 | Update PROOF.md SOTA section, HANDOFF.md, PROGRESS.md; flip EVAL_REPORT_SOTA off DRAFT only if the verdict checklist genuinely fills | **blocked (scope)** — handoff text below | `docs/EVAL_REPORT_SOTA.md` is NOT in DRAFT (status line already reads "contract-v2 results final … seed-11 and seed-23 replications verified under contract v2"); its verdict checklist still has one open item: *"status line in PROOF.md updated with final scope"*. PROOF.md / HANDOFF.md / PROGRESS.md live under `.dsh-24x7/` (outside this lane's write scope). Draft text for those files is in §P0.7 below for the Lead to apply. No DRAFT flip was performed or needed. |

## §P0.1 — Fleet poll / respawn / stderr capture runbook (Windows host)

Conventions are AGENTS.md load-bearing: PowerShell only; WMI spawn via
`Invoke-CimMethod -ClassName Win32_Process -MethodName Create` (never
`Start-Process`); payload wrapped `cmd /c "... 1> stdout.log 2> stderr.log"`;
durable paths `.dsh-24x7\`, `data\models\`, `data\raw\sources\`; jobs pin
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`.

1. Poll liveness + heartbeat (non-destructive):
   `powershell -File scripts\fleet_watchdog.ps1 -MaxRespawns 3` — writes
   `.dsh-24x7\fleet_heartbeat.json` with per-job
   `alive/pid/out_exists/out_stale/out_last_write/respawn_count/action`.
2. Respawn dead shards through the parametrized launcher only:
   `powershell -File scripts\fleet_spawn.ps1 -Manifest <jobs.json>` where
   `<jobs.json>` is regenerated by `scripts\fleet_manifest_sota.ps1` (schema
   `{workdir, logroot?, jobs:[{name, command, env?, out?}]}`; `workdir =
   "D:\dipcatcher"`). The launcher writes `spawn_receipt.json` per launch.
3. Stderr capture for shards that die twice: the watchdog's
   `respawn_count >= 2` entries must have their `stderr.log` copied into
   `.dsh-24x7\eval-full\<job>.stderr.2nd.log` (the `cmd /c` wrapper already
   splits `stdout.log`/`stderr.log` per job).
4. Post-run verification: heartbeat shows `alive: true` or
   `action: dead_exhausted_respawns` (never silent); every job has
   non-empty `stdout.log`; every twice-dead shard has its captured stderr.
5. Exact receipts that must result: an updated
   `.dsh-24x7\fleet_heartbeat.json` (hash-pinned into
   `quality/p0_evidence_inventory.json` at the next inventory regen) and
   `spawn_receipt.json` from each respawn.

## §P0.2 — TimesFM contract-v2 splice runbook

1. Validate shard pairs without writing:
   `uv run python scripts/splice_timesfm_fix.py --check-only --left <pre> --right <tfmfix>`
   — asserts row counts, asset identity (`X-4h` relabeled to `X`),
   `bars_sha256` equality and the protocol tuple
   `origins_per_asset/samples_per_origin/lookback/window/garch_window/seed/taus`.
2. Splice (writes `<stem><suffix>`, replacing ONLY the `timesfm` column):
   same command without `--check-only`.
3. Post-run verification: re-run `--check-only` on the spliced output against
   the pre-splice original and confirm all non-`timesfm` columns are
   bit-identical; archive the pre-splice matrix beside the output.
4. Exact receipt: a splice evidence JSON in the
   `.dsh-24x7/evidence-sota-tfmfix-probe.json` mold, with the pre/post
   `losses_sha256` and per-column identity assertions.

## §P0.3 — Contract-v2 re-merge runbook + verification

1. `uv run python scripts/sota_eval_native.py --merge-parts <parts> --bars-root data/raw/sources ...`
   (or the merge entry point used for `merge_*_v2aug.json`) with the
   contract-v2 scoring settings; bootstrap at `n_boot = 2000`.
2. Verification step P0.3.v (this session did NOT run it): assert
   `config.n_boot == 2000` and `scoring_contract` is contract v2 inside each
   merged receipt before citing it in `docs/EVAL_REPORT_SOTA.md` §4.1.
3. Exact receipts: `merge_<cell>_v2aug.json` + `.losses.npz` per cell, with
   `source_parts_sha256` binding every shard.

## §P0.6.v — Student-t 0-NaN coverage re-verification

Recompute per-model `emitted/finite_crps/missing_or_failed` from
`.dsh-24x7/evidence-sota-eval-h4f-v2.losses.npz` and diff against the
`coverage` block of `evidence-sota-eval-h4f-v2.json` (1500 origins, 5 assets ×
300). The claim "hardened fitter: 0 NaN over all 1500 origins" is RECEIPTED by
that coverage block + `.dsh-24x7/merge_h4fix_aug2.json`; treat any drift as a
failed claim and revert §4.2 wording rather than adjusting numbers.

## §P0.7 — Handoff text for `.dsh-24x7/` (Lead applies; out of this lane's scope)

For `PROOF.md` (SOTA status line): *"SOTA eval scope final under contract v2:
v3-d1 (TimesFM-v2 spliced), v4-d1, h4f full-coverage, native d1/h4 crossover,
seed-11/seed-23 corrected-pairing replications — receipts in `.dsh-24x7/`
(`evidence-sota-eval-*`, `merge_*_aug*.json`, `native/MERGED_*.json`,
`merge_s11_v2aug.json`, `merge_s23_v2aug.json`). No live-P&L claim."*
For `HANDOFF.md`: carry the P0.1–P0.7 table above verbatim.
For `PROGRESS.md`: *"P0 evidence debt: P0.4/P0.5 closeable-now (receipts
merged + cited); P0.3/P0.6 merged receipts present, row-level re-verification
runbooked (§P0.3.v/§P0.6.v); P0.1/P0.2 fleet + deep-cell splices runbook-ready
but blocked on remote execution; P0.7 PROOF/HANDOFF/PROGRESS updates pending
Lead application of this text."*

## Scope notes (for the verdict checklist)

- `docs/evidence/index.md` regeneration is Lead-owned (three lanes mint
  receipts); `receipts/p0_evidence_inventory_v1.json` was reported to the Lead.
- Nothing in this file flips `docs/EVAL_REPORT_SOTA.md` off any draft-like
  claim; its status line was already final and remains unchanged.
