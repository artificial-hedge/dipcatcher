# Sequential inference suite

The repo's research lanes all terminate in receipts; this family of
modules makes the *act of reading them over time* statistically honest.
Batch tests rerun on accumulating evidence silently inflate the
false-positive rate — every module here is built on **e-processes** or
**alpha-investing**, which stay valid under arbitrary stopping.

## The e-process core

`research/evalues.py` — `LossEProcess`: a betting e-process over a
per-origin proper-loss difference stream (challenger minus incumbent).
`E_t = Π(1 − λ·g_i)` with clipped predictable bets is a non-negative
test martingale, so Ville's inequality gives `P(ever E ≥ 1/α) ≤ α` —
a promotion decision may be taken at the first origin where `E` crosses
`1/α` without any correction for having peeked. `promotion_report()`
emits `evalue_promotion.v1` (deep-verified by `verify-receipt`).

## Race: sequential elimination

`research/fleet_race.py` + `dipcatcher race` — runs *two* e-processes per
head against a fixed chunk-0 incumbent: a promote process (head beats
incumbent) and a demote process (incumbent beats head). Heads are
eliminated or promoted with anytime-valid evidence; the winner is the
first promoted head or the last survivor (`verdict: anytime|final_mean`).
Sealed `fleet_race.v1` receipts.

## Corpus-level and online multiple testing

- `research/corpus_inference.py` — `corpus_audit` harvests every p-value
  and e-value committed under `receipts/`, pools the p-values under one
  BH-FDR bound, and merges e-values by product (Shafer's rule). Answers:
  *which artifact claims survive family-level correction?*
- `research/online_fdr.py` — Foster–Stine alpha-investing (LORD): a
  wealth-budget controller over an *ordered stream* of tests, so each
  new receipt's findings are tested as they arrive with mFDR ≤ level at
  every arrival time — no pool-and-rerun.

## Selection bias and stability

- `research/winner_curse.py` — the tournament winner's reported score is
  optimistically biased by the act of selection. Paired-bootstrap
  optimism correction + split-half honest control + selection-aware
  CIs; emits `winner_curse.v1`.
- `research/drift_alarm.py` — `EProcessDriftAlarm` fires on a *level
  shift* in a loss stream with P(ever false-alarming) ≤ α (a head whose
  persistent level exists from step 1 is absorbed into its baseline —
  the contract is change detection, not badness detection).
  `PageHinkleyAlarm` rides alongside as the classical CUSUM diagnostic.
  Emits `drift_alarm.v1`.

## The composite verdict

`research/honest_verdict.py` — `honest_verdict(scores)` composes all
three lenses into one sealed `honest_verdict.v1` claim:

| verdict | meaning |
|---|---|
| `confirmed` | bias corrected, e-process promoted at α, no drift |
| `supported_with_caveats` | edge holds; evidence not yet decisive or PH-only alarm |
| `not_supported` | corrected score erases the edge, or drift alarm fired |
| `inconclusive` | inputs missing/mismatched or a component lane unavailable |

Components are lazily imported and degrade gracefully — a missing lane
is named in `unavailable_lanes` and forces `inconclusive`, never a
silent pass.

## Running the suite

`research/verdict_run.py` replays the fleet tournament keeping the
per-origin streams the aggregates discard, then composites the verdict —
`dipcatcher verdict` writes a sealed `honest_verdict.v1` receipt.

`research/monitor_run.py` (`dipcatcher fleet-monitor`) streams every
(shard, head) cell through the whole anytime-valid family — coverage
breach rate, nested tail depth, PIT calibration, conformal
exchangeability, loss drift vs the fleet median — and pools the lane
e-values per cell via `emerge_mean` (valid under arbitrary dependence).
Lanes not yet merged report `lane_missing` on the `monitor_run.v1`
receipt rather than failing silently.

`research/lane_power.py` (`dipcatcher lane-power`) is the suite's own
capability bench: measured alarm rate and time-to-alarm per lane per
injected defect size, with the defect=0 row bounding the false-alarm
rate by α — the sequential claim is only useful if the lanes actually
fire.

`research/suite_health.py` (`dipcatcher suite-health`) audits the whole
evidence trail in one pass: every receipt under `--receipts-dir` is
re-verified (seal + kind contract), harvestable p-values/e-values are
pooled via `emerge_mean` (valid under the arbitrary dependence between
receipts), and a single corrupt artifact withholds the pooled claim —
the `suite_health.v1` receipt is itself sealed and verifiable.

### Self-drill evidence

`receipts/monitor_run_drill_clean.json` and
`receipts/monitor_run_drill_defect.json` are committed `monitor_run.v1`
receipts from a six-lane live pass over `iid_gaussian`/`regime_switch`
shards. Clean run: the only alarm is `qar` on `regime_switch` — a real
calibration miss the union drill detected (pooled e ≈ 1.7×10⁵). Defect
run: an injected biased quantile head fires coverage, tail, and
calibration on both shards (pooled e ≈ 9.9×10¹⁶) while drift correctly
stays silent on `regime_switch` — a persistent-from-start defect is
absorbed by the level-shift contract, not an alarm.

`receipts/lane_power_drill.json` is the suite's measured power curve:
nine lanes × defect ∈ {0, 0.5, 1.0} × 8 seeds at n=200. Every lane's
defect-0 alarm rate is 0/8 (within the α bound); calibration,
changepoint, loss-CS and promotion all fire at defect 0.5, and the
slower Bernoulli lanes (coverage, tail, conformal, drift) fire by
defect 1.0 — the conservative-null / real-power asymmetry is the
signature of valid e-processes, not a bug.

Two failure modes the drill surfaced and fixed:

- `changepoint_localize` alarmed on 25% of *stationary* streams — the
  per-split e-value threshold was applied to a max over ~75 candidate
  splits. The threshold is now scan-corrected (`log(n_candidates/α)`,
  Bonferroni over the overlapping windows): null alarm 0/40, detection
  of a planted 1.5σ shift preserved.
- All three Bernoulli lanes (`coverage_watch`, `coverage_cs`,
  `tail_watch`) coerced `None`/`2`/`NaN` into a non-breach — a missing
  observation silently deflated the measured breach rate. They now take
  a strict {bool, 0/1 int} flag and raise on anything else.

`test_committed_sealed_receipts_reject_tampering` mutation-tests the
committed corpus itself: every sealed `receipts/*.json` is perturbed
five ways (seal flip, seal removal, kind/schema rename, claim-digit
edit, top-level drop) and must come back invalid — zero survive.

`receipts/monitor_run_real_drill.json` is the suite's first non-synthetic
evidence: the six-lane monitor over 2,693-day Yahoo EOD return series
(8 symbols × 8 fleet heads, sealed with `data_label: yahoo_eod` — the
receipt label is derived from shard configs, never hard-coded). The
lanes discriminate correctly on real data: the calibration e-process
fires on nearly every (symbol, unconditional-head) cell — a static
512-day window cannot track vol clustering — while the conformal
martingale never alarms and the drift lane separates only the heads
that degrade relative to the fleet median.

`receipts/honest_verdict_real_drill.json` carries the composite verdict
on the same tape (`scripts/verdict_real_drill.py`): **not_supported**.
On real fat-tailed returns the gaussian head wins the raw tournament,
but the promotion e-process finds no evidence it beats the empirical
runner-up (E ≈ 0), calibration fires nearly everywhere, and Page–Hinkley
flags the regime structure the drift e-process correctly cannot confirm —
a genuinely honest negative claim, sealed. The bench inputs now carry the
same provenance gate: `SyntheticBook`/`CrossSectionalPanel` constructors
must declare `data_label`, and a mixed corpus fails closed rather than
inheriting a hard-coded SYNTHETIC stamp.

## Verifier coverage

Every kind above has a contract check in `research/evalue_contracts.py`
dispatched by `verify-receipt` on the `kind` tag — each embedded claim
is re-derived from the payload (e.g. `anytime_p == min(1, 1/E)`,
`corpus_reject_at_alpha ⇔ E ≥ 1/q`), so a tampered verdict fails
closed even under a valid reseal.

## References

Ville (1939); Ramdas, Ruf, Larsson, Koning (2020) betting e-processes;
Waudby-Smith & Ramdas (2024) anytime-valid confidence sequences;
Javanmard & Montanari (2018) LORD; Foster & Stine (2008) alpha-investing;
Fithian, Sun & Taylor (2014) post-selection inference.
