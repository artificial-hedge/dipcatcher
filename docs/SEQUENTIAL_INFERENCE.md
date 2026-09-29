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
