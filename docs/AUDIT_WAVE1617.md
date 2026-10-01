# Wave-16/17 module audit

Dedicated audit of the seven modules that landed post-pin in waves 16–17
(waived in the audit manifest pending this pass). All seven lift to
`audited`.

## Verdicts

| Module | Suite | Verdict | Notes |
|---|---|---|---|
| `metrics/adaptive_eps.py` | `tests/unit/core/test_adaptive_eps.py` | audited | e-PS (Lin, Ma, Ren & Wei 2026, arXiv:2609.26651) — posterior-sampled adaptive multiple testing with e-BH. Fail-closed on non-finite log increments (e>0 a.s. contract), out-of-range arms, nesting violations (`mark_rejected` rejects revoked rejections), all-rejected select. Seeded determinism via `SeedSequence.spawn` — data stream and posterior draws on independent children. `_inc_sum`/`_inc_sumsq` bookkeeping duplicates `_log_e` but feeds the variance proxy — harmless. exp() clipped at 1e300 for e-BH safety. Bench output self-labels SYNTHETIC; FDR/FDP/TPR proper scores only. |
| `metrics/conformal_oce.py` | `tests/unit/core/test_conformal_oce.py` | audited | OCE risk control (Farzaneh & Simeone 2026, arXiv:2608.28179). Fail-closed: uncertified grids return `certified=False` (no deployment without certificate), LP solver errors raise `RuntimeError`, all distribution/prob arrays validated for finiteness/sum-to-one. Entropic OCE pinned identical to `metrics.entropic_risk` (test-pinned). UCB uses per-reserve penalty range — a documented tightening of the paper's uniform [0,B]. |
| `models/diffusion_forecaster.py` | `tests/unit/models/test_diffusion_forecaster.py` | audited | DiffPTS (arXiv DDPM forecaster). `torch` import failure → informative `ImportError`. Both numpy and torch paths seed every draw (`np.random.default_rng(seed)`, `torch.manual_seed`); GPU determinism documented as not guaranteed. Shape/dtype validation on every public entry. |
| `models/greek_neutral_portfolios.py` | `tests/unit/models/test_greek_neutral_portfolios.py` | audited | Greek-neutral portfolio construction (arXiv:2609.33767). The Sharpe-like `performance_objective` (paper §5.2) is correctly namespaced `sim_internal_perf_objective_by_alpha` in outputs — never a headline metric — and fails closed on <2 observations, non-finite inputs, or degenerate variance. Torch fallback mirrors the numpy path. |
| `research/benches_w16.py` | `tests/unit/research/test_benches_w16.py` | audited | Wave-16 bench adapter — thin float-only mapping over `bench_ghcp` et al. Typed `except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError) → {}` is the established lane-absence convention: a lane that cannot run contributes no keys rather than fabricating zeros; non-finite maps are dropped explicitly before return. Every lane stamps SYNTHETIC and cites its paper. |
| `fx1 eval/rubric_banks.py` | `tests/fx1/test_rubric_eval.py` | audited | FinAutoRubric task-bank port. Bait criteria are intentional refusal probes (`hr-refuse-sharpe-headline` demands a forbidden headline — the correct answer is refusal); seeded rng; pydantic-validated criteria schema. |
| `fx1 eval/rubric_eval.py` | `tests/fx1/test_rubric_eval.py` | audited | Rubric generate→review→grade loop. The code layer beats an agreeing LLM reviewer: `_forbidden_demand_failures` rejects bait criteria even when the reviewer approves; writer/reviewer exceptions escalate to human review (flagged, excluded from grading) via `except Exception` + `# noqa: BLE001` — degrade-never-crash by design, counted under the except-Exception ratchet. `validate_fx1_output` runs on every model output; artifacts carry `RUBRIC_EVAL_LABEL`. |

## Manifest changes

- `quality/audit_coverage.json`: five module entries flipped
  `waived → audited` (doc: this page); `metrics`, `models`, `research`
  `n_modules` bumped to the actual recursive `.py` counts (waived files no
  longer excluded).
- `quality/audit_coverage_fx1.json`: `eval/rubric_banks.py` and
  `eval/rubric_eval.py` flipped `waived → audited`; `eval.n_modules` bumped.
