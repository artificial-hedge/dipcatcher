# SOTA 11 — Testing strategy for numerical and financial code

Status: research + audit note. No code changed by this document.
Date: 2026-09-28. Author: testing-strategy lane (web-researched, in-tree audited).
Repo state audited: `main` @ `3dafeb7`, plus the **uncommitted working tree**,
which diverges materially from HEAD in `pyproject.toml` / `Makefile` (§4.6).

All numbers below were measured from the tracked tree, not estimated. §8
gives the exact commands so any reviewer can reproduce them.

**Honesty framing.** This document is a correctness-engineering note. Nothing
in it is a market result, a performance claim, or evidence of profitability.
Every property test proposed here runs on **SYNTHETIC** inputs and must keep
the module-level `RESEARCH_ONLY = True` / `LIVE_PNL_CLAIM = False` constants
already used in `tests/property/` and `tests/unit/research/`. The proposed
tests check *mathematical identities of estimators*, never that a strategy
works. `FORBIDDEN_RESEARCH_METRIC_KEYS` (`quant_fund.research.catalog`) and
its fx-1 mirror `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` are themselves
invariants of the harness and are treated as protected surface in §6.3.

---

## 1. Verdict up front

The harness is **strong at the system layer and thin at the estimator layer**.

- **Strong:** end-to-end identity testing of money paths is close to
  state of the art for a research codebase. Byte-identical replay
  (`tests/property/test_fast_replay_byte_identity.py`), TLC model-checking of
  the order lifecycle plus Z3 proofs of accounting closed forms
  (`tests/formal/`, `docs/FORMAL_VERIFICATION.md`), a 3,904-line order-book
  invariant file with 226 `@given` sites, bitemporal PIT stateful testing
  (`tests/property/test_pit_asof_never_future.py`), fault injection on
  untrusted vendor data (`tests/unit/data/test_ingest_fault_injection.py`),
  and a real mutation-score record (`tests/property/mutation_scores.json`).
- **Thin:** of **58** modules in `src/quant_fund/metrics/`, only **4**
  (`returns`, `analytics`, `overfitting`, `cross_section`) have any
  Hypothesis coverage — **≈7%**. The modules the honesty contract names
  first — `scoring` (pinball, CRPS, QLIKE, PIT, Fissler–Ziegel),
  `probability` (Brier, log-loss, ECE, Kupiec, Christoffersen,
  Acerbi–Székely), `risk` (VaR/ES), `energy_score`, `density_forecast`,
  `var_backtest`, `es_backtest` — have **zero** generated-input coverage.
  They are covered instead by hand-computed known-answer tests, which are
  good at pinning formulas and structurally unable to find the class of bug
  that matters most in this codebase: an estimator that is right on the
  author's example and wrong on the boundary the author did not think of.
- **Unused leverage:** `hypothesis.extra.numpy` is imported in **0** files.
  Every numeric property test hand-rolls `st.lists(st.floats(...)).map(np.asarray)`,
  which forfeits dtype/shape/NaN/inf/subnormal generation for free.
- **Live regression in the working tree:** `pyproject.toml` has been reverted
  to an older `dipcatcher`-named variant that drops four registered markers,
  drops `pytest-xdist` / `pytest-split` / `pytest-benchmark` from the dev
  group, lowers the coverage floor from 80 to 70, and narrows `testpaths`.
  With that file in place, `--strict-markers` **fails collection** of
  `native`/`smoke`/`perf_full` tests. See §4.6. This is the single
  highest-priority finding in this audit and is independent of everything else
  in this document.

---

## 2. Technique summaries (with citations)

### 2.1 Property-based testing with Hypothesis

Random *generation plus shrinking* against a stated invariant, rather than a
fixed list of examples. Origin: QuickCheck [Claessen & Hughes, ICFP 2000];
Python implementation and the shrinking/database machinery: [MacIver, Eide &
Grey, *Hypothesis: A New Approach to Software Testing*, JOSS 4(37):1891,
2019].

What actually makes PBT pay off in numerical code, in descending order of
bug yield:

1. **Structural identities that need no oracle.** Non-negativity, monotonicity,
   idempotence, round-trip, invariance/equivariance, additivity, bounds
   containment. These are the metamorphic relations of §2.2 expressed as
   single-function properties.
2. **Differential (metamorphic) tests against a second implementation.** A
   slow reference or a higher-precision recomputation is a legitimate oracle
   [Higham, *Testing Linear Algebra Software*]. This repo already does exactly
   this for the Rust kernels (`tests/property/test_native_kernels.py` asserts
   `quant_core` ≡ NumPy reference, bit-exact or within `RTOL`/`ATOL`).
3. **Totality / fail-closed contracts.** "Never crashes, only raises or
   returns an honest NaN" is itself a property, and it is the one that maps
   directly onto this repo's fail-closed doctrine. `tests/fx1/test_properties.py`
   uses precisely this shape for the honesty validator.
4. **Stateful testing** (`RuleBasedStateMachine`) for anything with sequence
   dependence: broker cash, PIT vaults, ledgers. Three files already use it;
   `tests/formal/test_stateful.py` drives `SimulatedBroker` against `Account`
   and the TLA+-aligned trace validator.

Numerics-specific strategy guidance [Hypothesis numpy extra docs; NumPy
testing support]:

- Use `hypothesis.extra.numpy.arrays(dtype, shape, elements=..., fill=...)`
  instead of `st.lists(...).map(np.asarray)`. `fill` matters for large arrays
  (Hypothesis is optimized for small examples); `unique=True` gives distinct
  elements; `floating_dtypes()` covers 16/32/64-bit.
- **Test the non-finite frontier deliberately, then exclude it deliberately.**
  `st.floats()` includes NaN/±inf by default. For estimators that are
  documented to mask non-finite input "honestly" (this repo's convention),
  the right property is *two-sided*: finite inputs → finite output in range;
  non-finite inputs → either honest NaN or `ValueError`, never a fabricated
  number. `tests/property/test_adversarial_returns_and_sizing.py::test_nonfinite_returns_do_not_become_a_number`
  is the in-house exemplar.
- Keep generated data deterministic under a seeded `np.random.default_rng(seed)`
  with `seed` itself generated — this preserves shrinkability while making
  statistical assertions reproducible (`tests/property/test_cross_sectional_rankic.py`
  does this correctly).
- Bound example budgets explicitly. The repo's `adversarial_settings()`
  (20 examples CI / 80 nightly, `deadline=None`, `derandomize=True` in CI,
  `print_blob=True`) is the right pattern; `print_blob` is what lets a CI
  failure be replayed locally, and it appears in exactly **1** file.
- Persist failing examples. `.hypothesis/` currently holds 1,076 files and is
  self-gitignored by Hypothesis; `database=None` in the adversarial profiles
  deliberately turns that off for CI determinism. That is a defensible choice,
  but it means CI-found counterexamples survive only as the printed blob.

### 2.2 Metamorphic testing

When no oracle exists for a single input, assert a *relation* between the
outputs of several related inputs. Canonical survey: [Chen, Kuo, Liu, Poon,
Towey, Tse & Zhou, *Metamorphic Testing: A Review of Challenges and
Opportunities*, ACM Computing Surveys 51(1):4, 2018, doi:10.1145/3143561];
complementary surveys: [Segura, Fraser, Sánchez & Ruiz-Cortés, *A Survey on
Metamorphic Testing*, IEEE TSE 42(9):805–824, 2016]; [Chen et al., *Metamorphic
testing 20 years later*, ICSE 2018 companion, doi:10.1145/3183440.3183468];
relation-generation state of the art: [*Metamorphic Relation Generation*,
arXiv:2406.05397]. For scientific software specifically: [Kanewala, Bieman &
Ben-Hur, *Predicting metamorphic relations for testing scientific software*,
STVR 26(3):245–269, 2016].

The relation families that transfer directly to this repo (names follow the
MR-generation taxonomy: construction, arithmetic, data-relation, inference):

| MR family | Relation | Where it applies here |
|---|---|---|
| **Permutation** | Reordering cross-sectional members leaves a rank/covariance statistic unchanged; permuting ensemble members leaves CRPS/energy score unchanged | `metrics/cross_section`, `metrics/scoring::crps_*`, `metrics/energy_score`, `models/covariance` |
| **Translation / scale** | Positive scaling of returns scales vol/CRPS/QLIKE predictably and leaves Sharpe-like ratios invariant; adding a constant to all observations shifts the mean but not the spread | `metrics/returns`, `metrics/scoring`, `metrics/risk` |
| **Restriction / extension** | A statistic on a subsample equals the same statistic computed on the full sample masked to that subsample | `metrics/scoring::coverage`, `overlap_aware_qlike`, `date_level_equal_weight` |
| **Idempotence / absorption** | `rearrange_quantiles` applied twice equals applied once; appending an already-dominated observation does not move a max/drawdown functional | `metrics/scoring::rearrange_quantiles`, `metrics/returns::max_drawdown` |
| **Monotonicity** | A wider interval cannot reduce coverage; a larger tail mass cannot reduce ES; more trials cannot raise a deflated statistic | `metrics/scoring::coverage/interval_width`, `metrics/risk::historical_es`, `metrics/overfitting` |
| **Conservation** | Split + dividend conserves wealth; volume is conserved under resampling; cash is conserved across an order lifecycle | `data/corporate_actions`, `data/adapters/hf_ohlcv_1m`, `execution/simulated_broker` |
| **Causality / no-lookahead** | Restating data after time `t` leaves `asof(t)` byte-identical; a feature computed at `t` never reads `t+k` | `data/point_in_time`, `pit/`, `tests/property/test_adversarial_lookahead.py` |
| **Duality** | Two algebraically-equivalent routes to the same quantity agree within tolerance (quantile CRPS ↔ closed-form CRPS; energy score ↔ its U-statistic form) | `metrics/scoring`, `metrics/energy_score`, `native` ↔ reference |

Metamorphic relations are *not* weaker than oracles — for a proper scoring
rule they are frequently stronger, because propriety is itself a metamorphic
statement: the expected score is minimized at the true distribution.
`tests/unit/research/test_fissler_ziegel.py::test_fz0_properness_minimizes_at_true_es`
already exploits this (a Bugbot-caught regression where the un-scaled variant
minimized at `(1−α)·ES`). That test uses 400,000 seeded draws; it should be a
Hypothesis property over `alpha`, tail mass, and sample size.

### 2.3 Floating-point comparison and numerical-stability testing

Definitions to keep straight [Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., SIAM 2002; Higham, *What is Numerical Stability?*, 2020]:

- **Forward error** = ‖computed − exact‖. **Backward error** = smallest Δx with
  f(x + Δx) = computed. Forward ≤ κ(problem) × backward.
- **Backward stable**: small backward error for all inputs. **Forward stable**:
  forward error of the order a backward-stable method would achieve.
  Backward stable ⇒ forward stable; the converse fails (Gauss–Jordan).
- Testing consequence: where a backward-error analysis exists, **assert the
  bound**, not the digits. Where it does not, test forward error against a
  higher-precision recomputation [Higham, *Testing Linear Algebra Software*;
  LAPACK Working Note 41 — the LAPACK test suite is the canonical worked
  example, including its test-matrix generators].

Practical comparison rules:

- Prefer `np.testing.assert_allclose(actual, desired, rtol, atol)` and reserve
  `assert_array_max_ulp` / `assert_array_almost_equal_nulp` for cases where a
  digit-level bound is the actual contract. NumPy's own documentation
  deprecates `assert_almost_equal` / `assert_approx_equal` in favour of the
  three above [NumPy testing support].
  **This repo: 260 files use `pytest.approx`, 37 use `assert_allclose`, 0 use
  any ULP assertion.** The ULP tools are exactly what a "bit-for-bit or
  documented tolerance" claim should be expressed in.
- Choose `rtol` from the algorithm, not from a vibe. A rule of thumb:
  `rtol ≈ κ × n × u` where `u = 2⁻⁵³ ≈ 1.1e-16` is the unit roundoff, `n` the
  accumulation length, and `κ` the condition number of the sub-problem. For a
  well-conditioned O(10³)-length dot product that lands near `1e-12`; for an
  ill-conditioned covariance estimator no fixed `rtol` is meaningful and the
  test must be a backward-error or higher-precision comparison instead.
- `atol` exists to cover the neighbourhood of zero where relative error is
  undefined. A bare `assert_allclose(x, 0)` with default `atol=0` is a
  self-inflicted flake; always pair `atol` with the scale of the quantity.
- **Condition-number sensitivity is a testable property, not a comment.** For
  every estimator whose `κ` can blow up (Ledoit–Wolf shrinkage intensity,
  factor covariance, HAC bandwidth, DCC, Hessian-based standard errors in
  `utils/numeric::numeric_hessian` with its fixed `HESSIAN_STEP_SCALE = 1e-5`),
  the property to assert is *graceful degradation*: the output must remain
  finite and within its documented contract as κ grows, and must fail closed
  rather than silently return a plausible wrong number.

Compensated summation. Kahan/Neumaier compensated summation reduces the
roundoff constant from O(n) to O(1) but **does not** remove condition-number
dependence: the relative-error bound of any fixed-precision summation remains
proportional to the summation condition number Σ|xᵢ|/|Σxᵢ| [Kahan summation
algorithm, Wikipedia; Cook, *Summing an array of floating point numbers*,
2019]. CPython's built-in `sum()` has used Neumaier accumulation since 3.12;
NumPy gives **no** summation-order guarantee, though pairwise summation is
"often" used.

That last sentence is the reason `src/quant_fund/backtest/fast_replay.py`
carries explicit comments about reproducing CPython 3.12 compensated
summation and about `np.sum`'s pairwise reduction differing "at the last ulp"
— and the reason `tests/property/test_fast_replay_byte_identity.py` asserts
**Arrow IPC byte equality**, not float tolerance, between the vectorized and
event-loop engines. This is the single best-executed numerical-testing
decision in the repo and should be treated as the house standard for any
"fast path ≡ reference path" claim.

A stability test suite worth adding (all cheap, all SYNTHETIC):

1. **Cancellation battery.** Ill-conditioned sums (`[1e16, 1, −1e16]`,
   harmonic series with mixed signs, near-equal large magnitudes) against
   `math.fsum` / float128 / `Fraction` exact references. Any accumulator in
   the repo that must be bit-stable gets a named worst case.
2. **Order-independence probe.** For each reduction that claims
   order-independence, assert `f(x) ≈ f(x[perm])` within the documented `rtol`,
   and *record the achieved deviation* so a future NumPy pairwise-order change
   shows up as a diff rather than a mystery.
3. **Conditioning sweep.** Generate matrices with controlled κ
   (`st.floats` on the log-scale of target condition number, then build via
   SVD or a `xlatms`-style generator), assert the backward-error bound and
   assert the estimator raises or returns NaN rather than a finite lie.
4. **Overflow/underflow frontier.** Products and exponentials near
   `np.finfo(np.float64).max` / `.tiny`: `metrics/evalues` (products of
   likelihood ratios), `metrics/entropy::lempel_ziv_complexity`,
   `models/*` log-likelihoods. The invariant is `inf`/`0.0`/NaN must be
   reachable-but-labelled, never silently clamped into range.

### 2.4 Golden-master / snapshot testing

Record an approved output; later runs must reproduce it. Two distinct uses,
and conflating them is the main failure mode:

- **Characterization / regression** ("did behaviour change?") — legitimate and
  valuable for legacy numeric code with no oracle.
- **Correctness** ("is this right?") — *never* established by a snapshot. A
  snapshot can only prove stability.

Tools: [syrupy](https://syrupy-project.github.io/syrupy/) is the most popular
general pytest snapshot plugin (fails on missing snapshots, `--snapshot-update`
to refresh, xdist-aware) but has **no numeric tolerance support**; for arrays
and frames use [pytest-regtest](https://pytest-regtest.readthedocs.io/en/stable/),
whose `snapshot.check` ships handlers for NumPy arrays, pandas and **polars**
frames with explicit `rtol`/`atol` (both default to `0.0`, which the docs
correctly call "often too prudent").

This repo already has four flavours of golden master, and they are better than
most because they are *narrow*:

| Artifact | Kind | Notes |
|---|---|---|
| `tests/unit/public_api_snapshot.txt` | API surface lock | Diffed by `tests/unit/test_public_api.py`; catches accidental signature/export drift |
| `tests/regression/test_scoring_frozen.py` | Formula pin | `pinball_loss(y=0,q=−1,τ=0.05) == 0.05`; `qlike(e,1) == e−2`. Header says "do not 'improve' without updating MATH_SPEC" — exactly right |
| `tests/perf/baselines/baseline.json` + `check_regression.py` | Performance golden master | **Calibration-normalized**: ratio to `test_calibration_matmul`, threshold 0.50. Absolute ms is not comparable across runners; this is the correct design |
| `tests/property/test_fast_replay_byte_identity.py` | Byte-exact golden master | Arrow IPC bytes, not floats |

Gaps: no snapshot for **receipt/notebook JSON shape** beyond the inline
field-presence assertions in `ci.yml`'s smoke job; no snapshot for the
**research scorecard family set** (`REQUIRED_BENCHMARK_FAMILIES`) as data; no
snapshot of `verify-research` output structure. All three are cheap and all
three currently fail only via bespoke `python - <<'PY'` heredocs embedded in
CI YAML, which is untestable locally and invisible to coverage.

Anti-patterns to keep avoiding: snapshotting a whole notebook blob (every
legitimate change becomes a wall of red, and reviewers learn to
`--snapshot-update` reflexively); snapshotting floats without `rtol` (platform
flake); snapshotting anything that embeds a timestamp, run id, or absolute path
without normalization.

### 2.5 Mutation testing

Seed small syntactic faults; a test suite that keeps passing is not checking
anything. Survey: [Jia & Harman, *An Analysis and Survey of the Development of
Mutation Testing*, IEEE TSE 37(5):649–678, 2011]; operator-sufficiency basis:
[Offutt, Lee, Rothermel, Untch & Zapf, *An experimental determination of
sufficient mutant operators*, ACM TOSEM 5(2):99–118, 1996].

Industrial practice — the part that determines whether a gate survives contact
with a team:

- [Petrović & Ivanković, *State of Mutation Testing at Google*, ICSE-SEIP 2018,
  pp. 163–171]: **diff-scoped**, probabilistic, and *arid*-line suppression.
  ~30% of covered diffs, surfaced in code review rather than as a build
  failure. This is the design that scaled to 6,000 engineers.
- [Petrović, Ivanković, Kurtz, Ammann & Just, *Practical Mutation Testing at
  Scale: A View from Google*, IEEE TSE 2021, doi:10.1109/TSE.2021.3107634]:
  precision matters more than score; developers need the surviving mutant's
  diff, not a percentage.
- Practitioner consensus (CircleCI, mutmut guides, `mutation-gate`): **do not
  gate on a repo-wide percentage.** Gate on "no new surviving mutants in
  protected modules" and run diagnostic-first for a few weeks before making it
  blocking. A hard 100% target invites over-specified tests that pin
  implementation detail.

Evidence that mutation score is the right *signal* and the wrong *KPI*:
[Just, Jalali & Ernst, *Are Mutants a Valid Substitute for Real Faults in
Software Testing?*, FSE 2014] found statistically significant correlation with
real-fault detection **independent of coverage**; [Papadakis et al., *Are
Mutation Scores Correlated with Real Fault Detection?*, ICSE 2018,
doi:10.1145/3180155.3183519] found the correlation is **weak once test-suite
size is controlled**, but that raising mutation score still significantly
improves fault detection versus randomly selected suites of equal size. Read
together: mutants are good *guidance for where to test harder*, not a number
to hit.

Python tooling: [mutmut](https://github.com/boxed/mutmut) (v3 mutates
functions, resumable, native test-impact selection via its trampoline —
which is exactly what `scripts/mutation_score.py` relies on) and
[Cosmic Ray](https://github.com/sixty-north/cosmic-ray) (plugin distributor
model, better for spreading work across workers). Diff-scoped pre-commit
pattern: [mutation-gate](https://github.com/mikemartincode/mutation-gate)
(triggers on radon cyclomatic complexity ≥ threshold, computes an AST-precise
blast radius, runs mutmut only on affected files).

### 2.6 Fuzzing data parsers

Distinct from PBT: PBT asserts *properties of structured inputs*; fuzzing
asserts *non-crash on unstructured/hostile bytes* and uses coverage feedback to
walk deep into the parser. Origin of the empirical case: [Miller, Fredriksen &
So, *An Empirical Study of the Reliability of UNIX Utilities*, CACM
33(12):32–44, 1990] — crashed >24% of ~90 utilities across 7 UNIX versions
with random input. Survey: [Manès, Han, Han, Cha, Egele, Schwartz & Woo, *The
Art, Science, and Engineering of Fuzzing*, IEEE TSE 47(11):2312–2331, 2021].
On the division of labour, see also [Goldstein's dissertation on PBT vs
fuzzing practice] and [Teodorescu, *Fuzzing vs property testing*, 2018]:
Hypothesis for domain models and readable properties; a coverage-guided fuzzer
for decoders, tokenizers and importers where malformed input is *expected*.

Python tooling: [Atheris](https://github.com/google/atheris) (libFuzzer-backed,
instruments Python bytecode, supports native CPython extensions and can pair
with ASan/UBSan; `FuzzedDataProvider` converts bytes to typed inputs). Its own
docs note the weakness: without a grammar, structured formats get rejected
early and coverage stays shallow — mitigations are custom mutators or
**driving Hypothesis from the fuzzer** (`atheris.instrument_all()` plus a
strategy-driven target reuses the existing generators and oracles and gains
coverage-guided search). Continuous integration: [ClusterFuzzLite]
(PR-scoped short runs + batch runs that grow a corpus + daily coverage reports
+ corpus pruning, OSS-Fuzz-compatible build).

Highest-value targets in this repo, all of which already have a documented
fail-closed contract to fuzz *against*:

1. `quant_fund.data.sources.normalize` (`csv_rows`, `normalize_observations`,
   `normalize_ohlcv`) — the docstring in
   `tests/unit/data/test_ingest_fault_injection.py` states the contract
   precisely: "every malformed OHLCV batch must fail CLOSED … never pass
   silently into a 'valid' frame." That is a fuzzing oracle: any uncaught
   exception other than `SourceError`, and any *accepted* frame that violates
   the OHLC envelope (`low ≤ open,close ≤ high`), is a real defect.
2. `quant_fund.data.point_in_time` (`require_pit_columns`,
   `validate_feature_frame`) — PIT gate bypass is a research-integrity failure,
   not just a crash.
3. `quant_fund.data.adapters.parquet` / `hf_ohlcv_1m` (`normalize_vendor_frame`,
   `resample_ohlcv`, `minute_gap_report`) — real vendor files are adversarial.
4. `quant_fund.utils.hashing` (`canonical_json_bytes`, `_json_sort_key`,
   `hash_file`) — canonicalization must be total and collision-resistant on
   hostile input; the mutation run already surfaced 19 equivalent survivors
   clustered in exactly this sort-key path (§4.5), which is a hint that the
   *behavioural* surface there is under-probed even though the mutants are
   equivalent.
5. Receipt / corpus JSON parsing (`research/receipt_schema.py`,
   `fx1` corpus loader, `paper/ledger.py`).

Seed corpora matter more than runtime [Atheris/ClusterFuzzLite docs; the Zebra
fuzzing programme used mainnet-derived public data as seeds]: commit a small
`tests/fuzz_corpus/` of *real-shaped but weird* fixtures — truncated parquet
footers, mixed-precision CSVs, non-UTF8 symbol strings, duplicated
`event_time`, `available_time < event_time`, absurd magnitudes, empty payloads.

### 2.7 Coverage adequacy beyond line coverage

Line/statement coverage answers "was this executed"; it says nothing about
"was this checked". The empirical record:

- [Ahmed, Gopinath et al., *Can Testedness be Effectively Measured?*, FSE
  2016]: both statement coverage and mutation score correlate only *weakly*
  negatively with future bug-fixes — but elements covered by *any* test see
  about half as many bug-fixes as uncovered ones. The signal is largely
  binary (covered vs not), not graded.
- [Inozemtseva & Holmes, *Coverage is not strongly correlated with test suite
  effectiveness*, ICSE 2014] and [Namin & Andrews, *The influence of size and
  coverage on test suite effectiveness*, ISSTA 2009]: coverage's explanatory
  power collapses once test-suite size is controlled.
- [*Mind the Gap: The Difference Between Coverage and Mutation Score Can Guide
  Testing Efforts*, arXiv:2309.02395]: mutation score and coverage correlate
  (r ≈ 0.75 overall) but diverge most exactly where it matters — well-covered
  files show large positive "oracle gaps", i.e. **code that is executed but
  poorly checked**. The coverage−mutation-score gap per file is a better
  work-list than either number alone.

Adequacy criteria worth tracking for this codebase, in order of
cost-effectiveness:

1. **Branch coverage** — already on (`[tool.coverage.run] branch = true`).
   Keep it; it is necessary and cheap.
2. **Mutation score, per module, on protected paths** — the only criterion
   that measures assertion quality. See §6.3.
3. **Property-density**: modules under a numerical-invariant contract should
   have ≥1 generated-input test per public estimator. Currently 4/58 (§4.3).
4. **Contract coverage, not line coverage, of fail-closed paths.** This repo's
   real risk is a guard that silently stops guarding. The existing
   `test_*_branch_killers.py` files (`test_cost_branch_killers.py`,
   `test_returns_branch_killers.py`, `test_hashing_branch_killers.py`) are a
   hand-rolled version of this and a good idea; mutation testing automates it.
5. **Per-package floors with ratchets** — already implemented and the right
   shape: global `fail_under = 80` (ratcheted 2026-09-25, measured 84.93%
   local / 81.51% CI-Linux) plus `[tool.proofcore.coverage-floors]`
   `pit = 90, proof = 90, leakage = 85, reality = 90, proofcore = 90`,
   enforced by `quant_fund.proofcore.ci coverage-gate`. The comment
   "Ratchet-only thereafter: raise, never lower" is the correct governance.

What **not** to do: raise the global line-coverage floor as a proxy for
quality. At 80%+ the marginal line is a defensive branch; the marginal *bug*
is in an assertion that passes either way.

---

## 3. What "adequate" looks like for this harness (target state)

| Layer | Criterion | Today | Target |
|---|---|---|---|
| Money paths (broker, ledger, replay) | Byte/state identity + model checking | Byte identity, TLC, Z3, stateful PBT | Hold. Add mutation run on `execution/costs.py` to CI |
| Estimators (`metrics/`) | ≥1 generated-input property per public estimator, from a written invariant catalog | 4/58 modules | ≥40/58, all honesty-contract modules first |
| Data engine | Metamorphic conservation + causality + fuzz non-crash | Resample conservation, PIT stateful, fault injection, calendar tiling | + corporate-action conservation as PBT, + Atheris harnesses on `sources/normalize` |
| Numeric stability | Documented `rtol`/backward-error bound per estimator; ULP assertions where bit-identity is the contract | `RTOL`/`ATOL` in `native.reference`; byte identity in fast replay | Stability battery (§2.3) + tolerance rationale in `MATH_SPEC.md` |
| Assertion quality | Mutation score recorded and ratcheted on protected modules | 3 modules, recorded manually, not in CI | CI job, ratcheted floors, no-regression gate |
| Coverage | Branch coverage + ratcheted per-package floors | Global 80 + 5 package floors | Hold; add `metrics` floor once property coverage lands |
| Reproducibility | Derandomized CI, pinned budgets, replayable blobs | `HYPOTHESIS_PROFILE=ci`, adversarial profiles | + `.hypothesis` example DB seeded for the property lane; `print_blob` everywhere |

---

## 4. Audit of the current suite

### 4.1 Topology

Tracked files under `tests/` (`.py`):

| Lane | Files | Notes |
|---|---|---|
| `tests/unit/` | 851 | 27 subpackages; the mass of the suite |
| `tests/property/` | 30 | Hypothesis lane; includes `_profiles.py`, `_books.py`, `mutation_scores.json` |
| `tests/regression/` | 15 | Golden masters, frozen formulas, sealed receipts |
| `tests/fx1/` | 27 | Separate lane, deliberately outside `testpaths` (ADR-0003); run via `make fx1-test` |
| `tests/perf/` | 5 | pytest-benchmark + calibration-normalized regression check |
| `tests/formal/` | 4 | TLC conformance, Z3 accounting proofs, stateful PBT |
| `tests/end_to_end/` | 3 | SYNTHETIC pipeline, proofcore smoke |
| `tests/native/` | 1 | Rust `quant_core` bench |
| `tests/examples/` | 1 | Examples gallery runs |
| `tests/support/`, `tests/fixtures/`, `tests/leakage_fixtures/` | — | Shared synthetic fixtures; session caches |

Totals: **975** tracked test files, **~7,502** `def test_` functions (of which
**248** in `tests/fx1`). **42** files import Hypothesis, **39** contain
`@given`, **~358** `@given` sites — **226 of them in one file**
(`test_order_book_metrics_identity.py`). Excluding that file, the property
lane has ~132 generated-input tests across 38 files. That is a real but narrow
investment, and its density is very unevenly distributed.

Dead weight found: `tests/tests/` holds **233 untracked `.py` files** (a
restored copy of the dropped mirror; `git ls-files tests/tests` returns 0).
AGENTS.md says the `tests/tests` mirror was dropped and any stray must stay out
of default collection — it currently does, because HEAD `testpaths` enumerates
`tests/unit`, `tests/property`, `tests/regression`, `tests/end_to_end`,
`tests/formal` rather than `tests`. But the *working-tree* `pyproject.toml`
sets `testpaths = ["tests"]`, which would collect all 233 strays. There is also
a `.backup-prerestore/mirrors/tests_tests/` tree. Cleaning both is a
prerequisite for any coverage number to mean anything.

### 4.2 Markers

Registered (HEAD `pyproject.toml`), with `--strict-markers` enforced:

| Marker | Registered description | Files using it | Gate behaviour |
|---|---|---|---|
| `network` | requires internet | 1 (`tests/unit/data/test_hf_ohlcv_1m.py`) | Excluded everywhere (`-m "not network"`) |
| `slow` | long tests; full suite on schedule/dispatch | 6 (+ node-id list) | Excluded from PR gate |
| `smoke` | collected by the CI smoke job | 6 | Selected by `smoke` job |
| `native` | quant_core parity, selected by rust-accel job | 3 | Selected by `rust-accel` job |
| `synthetic` | uses synthetic market data | 22 | Informational |
| `perf_full` | larger synthetic benchmarks | 3 | Excluded from the perf CI subset |

Assessment: the marker set is well-designed and honest about intent. Two
observations.

1. **`synthetic` is drastically under-applied.** 22 files carry it, but the
   overwhelming majority of the ~975 test files run on synthetic data. Either
   apply it via a conftest hook (`pytest_collection_modifyitems` adding
   `synthetic` to any test in a module that declares `RESEARCH_ONLY = True`)
   or stop treating it as a per-file marker. As it stands it cannot be used to
   answer "which tests touch market-shaped data", which is the question the
   honesty contract cares about.
2. **`slow` is now expressed twice** — as a marker (6 files) and as a node-id
   list (`tests/slow_nodeids.txt`, currently comments only). The node-id
   mechanism is the better one (the conftest docstring explains why: paths are
   node ids, so moving a module without updating the list keeps the test in the
   fast gate — a fail-*loud* default). Since the list is empty, the PR gate
   today runs the entire offline suite; that is fine but means the marker-based
   `slow` set is the only actual exclusion, and the two mechanisms should be
   reconciled into one before either is extended.

### 4.3 Fixtures and shared infrastructure

- **`tests/conftest.py`** does four jobs: sys.path bootstrap, session-cache
  installation, node-id slow-marking, and fixture-timing telemetry
  (`pytest_fixture_setup` hookwrapper recording setups ≥0.25s, dumped per-xdist
  worker to `DIP_FIXTURE_LOG`). The per-worker split is correct — the
  controller does not run tests, so a single file would drop counts. It also
  registers the `ci` Hypothesis profile (`derandomize=True, max_examples=100`)
  when `HYPOTHESIS_PROFILE=ci`.
- **`tests/support/session_cache.py`** is the most sophisticated piece of test
  infrastructure in the repo: a source-stamped (`session-cache-v3`, sha256 over
  every `src/quant_fund/**/*.py` + `pyproject.toml` + `uv.lock`, truncated to
  20 hex), cross-worker, file-locked cache for `bench_northset` and
  `ensure_silver`/`build_gold`/`panel`, with **monkeypatch detection** (a
  fingerprint of module callables is compared to the one taken at install; any
  patch bypasses the cache), reentrant lock depth for the root→slot lock order,
  atomic temp-file-then-replace writes, and a `DONE` sentinel. It caches only
  `SYNTHETIC` configs. This is a genuinely careful design and it is the reason
  the suite is fast enough to shard 4× on every PR.
  - Residual risk worth naming: the cache *snapshots and restores lake
    artifacts*, so a cache hit can mask a bug in artifact layout. `_input_token`
    mitigates the worst case (a hand-edited lake is keyed by its own bytes and
    cannot be overwritten), but a test that asserts on lake *files* rather than
    on returned values is not safely cacheable. `DIP_DISABLE_SESSION_CACHE=1`
    is the escape hatch; it should be set in at least one scheduled CI job.
- **`tests/property/_profiles.py`** — `adversarial-ci` (20 examples,
  `deadline=None`, `derandomize=True`, `database=None`, `print_blob=True`,
  three health checks suppressed) and `adversarial-nightly` (80 examples,
  `derandomize=False`). Deliberately named so they do not clobber Hypothesis's
  built-in `ci` profile. Correct and well-documented.
- **`tests/property/_books.py`** — `research_config()` (relaxed research book,
  no broker, no live order path) + `daily_bars()`. Good; should be promoted to
  a shared fixture module so the metrics property tests in §6.1 can reuse it.
- **`tests/unit/research/explainability/conftest.py`** — planted-signal
  synthetic design (`y = X @ w + ε`) with a deterministic least-squares head.
  The planted-signal pattern is the right way to test attribution/PDP; it
  should be generalized into `tests/support/`.
- **`tests/unit/parity/conftest.py`** — shared SYNTHETIC tapes, explicitly
  labelled "Not market evidence".
- **`tests/unit/diffbacktest/conftest.py`** — pins JAX to CPU before import.
  Necessary and correctly placed.

Fixture-usage gaps: no `conftest.py` in `tests/property/`, so the property
lane has no shared strategy library. Every property file re-declares its own
`_ret = st.floats(min_value=-0.4, max_value=0.4, allow_nan=False, ...)` /
`_POSITIVE_FLOAT` / `_FINITE` / `_WEIRD`. This is duplicated domain knowledge
that drifts.

### 4.4 Parallelism, sharding, CI

- **PR gate** (`make test` / `ci.yml` `test` job): `-n auto --dist loadfile
  -m "not network and not slow"` plus pytest-split `--splits 4 --group N
  --durations-path .test_durations` across a `python × shard` matrix
  (3.12, 3.13) × 4. `--dist loadfile` is the right choice with the session
  cache (keeps same-file tests on one worker, maximizing cache hits).
- **Scheduled/dispatch**: full offline suite including `slow`.
- **Coverage**: sharded partials uploaded as artifacts, combined in a
  dedicated `coverage` job, single threshold from `pyproject` (no inline
  `--cov-fail-under` so CI and local cannot drift). Well done — the comment
  about not re-combining controller `.coverage` with worker fragments to avoid
  double-counting shows the failure mode was actually hit.
- **Dedicated property lanes**: `adversarial-properties.yml` (PR, `ci` profile,
  6 files, 20-min timeout) and `property-nightly.yml` (06:00 cron, `nightly`
  profile, same 6 files, 45-min timeout). **These two workflows run the same
  six files.** The other 24 property files ride only in the sharded `test` job.
  Consequence: the nightly budget increase (20→80 examples) applies to the
  *backtest/cost/receipt* lane but not to the calendar, PIT, LOB, order-book,
  overfitting, robustness, explainability, capacity, replay or reality
  properties — several of which are the most invariant-dense files in the repo.
- **`matrix.yml`**: cross-platform (Linux/macOS/Windows × 3.12–3.14),
  file-count-based sharding rather than xdist to avoid nested workers, with
  `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` pinned for cp1252 Windows runners.
  Correct and thoughtful.
- Other lanes: `fx1.yml` (lint + mypy + tests + honesty-inheritance gate +
  corpus contract smoke), `proofcore.yml`, `scorecard.yml`, `simtest.yml`,
  `codeql.yml`, `secret-scan.yml`, `dependency-review.yml`, `release.yml`.

### 4.5 Mutation testing — current state

`scripts/mutation_score.py` runs mutmut 3.8.0 against three hand-picked
modules with narrow pytest selections (so each mutant only reruns tests that
import it), writes a temporary `setup.cfg` (`[mutmut]` with `source_paths`,
`also_copy`, `pytest_add_cli_args_test_selection`, `process_isolation =
forkserver`, `use_git_change_detection = false`, `-m "not network"`), refuses
to overwrite an existing `setup.cfg`, cleans up in a `finally`, and records
survivor diffs when `MUTMUT_SURVIVOR_DIR` is set. `mutants/` is gitignored.
The score formula is documented in the JSON: `(killed + timeout) / (total − skipped)`.

Recorded results (`tests/property/mutation_scores.json`, `HYPOTHESIS_PROFILE=ci`,
NumPy 2.5.3):

| Module | Mutants | Killed | Survived | Timeout | Score | Wall time |
|---|---|---|---|---|---|---|
| `execution/costs.py` | 207 | 207 | 0 | 0 | **100.0%** | 48.8 s |
| `metrics/returns.py` | 364 | 325 | 39 | 0 | **89.3%** | 58.8 s |
| `utils/hashing.py` | 187 | 160 | 25 | 2 | **86.6%** | 55.3 s |

Every survivor is annotated with an equivalence argument, and the arguments are
specific and checkable rather than dismissive: `float('nan')`/`float('NAN')`
identity (15), `reshape(-2)` ≡ `reshape(-1)` (7), `reshape(None)` flattened by
a following `cumprod`/`std`/`mean` (5), `np.asarray([], dtype=None)` still
float64 (4), `cagr`'s `years <= 0` unreachable after a positive-period check
(1), `errstate invalid/divide` no-ops on the observed NumPy (4), `or`→`and` on
non-finite guards still yielding NaN (3); and in hashing, `hash_file` chunk
sizes that still hash the whole file (3), trailing-comma `getattr` identity
(3), and 19 `json.dumps` sort-key mutants argued equivalent with a 1,485-pair
probe. **This is the correct discipline** and it is better than most industrial
mutation records.

Gaps: (a) not wired into any CI workflow or Makefile target — the numbers are a
manual artifact and can silently go stale; (b) three modules out of 733 tracked
`src/` files; (c) no floor, no ratchet, no "no new survivors" check; (d) the
`hashing.py` equivalence cluster is so large (19 of 25) that the *behavioural*
surface of `_json_sort_key` is effectively unprobed by mutants, which argues
for targeted property tests there rather than more mutants.

### 4.6 Working-tree divergence (blocking finding)

`git diff HEAD -- pyproject.toml` shows the working tree reverting the project
to an older shape. Concretely, versus committed HEAD:

| Item | HEAD | Working tree | Effect |
|---|---|---|---|
| `[project].name` | `fx-1`, `dynamic = ["version"]` (from `src/fx1/__init__.py`) | `dipcatcher`, `version = "1.0.0"` | Contradicts `docs/FX1_API_STABILITY.md` ("`fx1.__version__` is canonical semver; pyproject reads it via hatch dynamic version") |
| Registered markers | 6 (`network, slow, smoke, native, synthetic, perf_full`) | 3 (`network, slow, synthetic`) | **`--strict-markers` fails collection.** Verified: `pytest tests/property/test_native_kernels.py --collect-only` → `ERROR … 'native' not found in 'markers' configuration option`. Same for every `smoke`/`perf_full` file |
| dev group | `pytest-benchmark`, `pytest-xdist`, `pytest-split`, `z3-solver`, … | dropped | `-n auto`, `--splits`, pytest-benchmark lanes, and the Z3 formal proofs cannot run |
| `testpaths` | `tests/unit, tests/property, tests/regression, tests/end_to_end, tests/formal` | `tests` | Would collect the 233 untracked `tests/tests/` strays and `tests/fx1`, breaking the deliberate lane separation (AGENTS.md, ADR-0003) |
| `[tool.coverage.run] source` | `["fx1", "quant_fund"]` | `["quant_fund"]` | fx-1 coverage silently unmeasured |
| `fail_under` | 80 (ratcheted, with measured evidence in the comment) | 70 | Ratchet broken |
| `[tool.proofcore.coverage-floors]` | present (90/90/85/90/90) | absent | Per-package gates lose their single source of truth |
| optional extras | `explainability` (shap, with a documented darwin-x86_64 pin fork), `jax` | absent | `diffbacktest` CI job cannot sync |
| scripts | `fx1`, `verify-ledger`, `mc-engine` | absent | CI smoke steps that invoke these fail |
| ruff | `C901` + `[tool.ruff.lint.mccabe] max-complexity = 74` | `C901` absent | `test_quality_ratchet.py`'s `MCCABE_CEILING` guard loses its config counterpart |
| mypy | `packages = ["fx1", "quant_fund"]`, `warn_return_any = true` globally, `mypy_path = "typings"`, 397-module strict allowlist | narrowed | Ratchets documented in `pyproject` comments and enforced by `scripts/check_mypy_strict_allowlist.py` become inconsistent |

`uv.lock` in the working tree also no longer contains `pytest-xdist`,
`pytest-split` or `pytest-benchmark`, so `uv sync --frozen` cannot restore them.
Since AGENTS.md states `uv.lock` is authoritative and `--frozen` must pass, and
since CI parity is an explicit gate, **the working tree cannot currently pass
the CI it describes**. Any adoption work in §6 should be sequenced after this
is resolved (restore HEAD's `pyproject.toml`/`uv.lock`, or land the intended
rename deliberately in one commit with `uv lock` regenerated).

### 4.7 Property-coverage gap in `metrics/` — the main technical finding

Of 58 modules, these 4 have Hypothesis coverage:

| Module | Property file(s) | Invariants asserted |
|---|---|---|
| `metrics/returns` | `test_adversarial_returns_and_sizing.py`, `test_drawdown.py`, `test_turnover_identity.py`, `test_native_kernels.py` | wealth = compound product; all-positive returns ⇒ zero drawdown; Sharpe scale-invariant and sign-antisymmetric; non-finite ⇒ NaN never a number; flat returns ⇒ zero CAGR/downside; turnover is an L1 pseudometric (identity, symmetry, triangle inequality); inverse-vol weights sum to 1 and are permutation-equivariant |
| `metrics/analytics` | `test_adversarial_receipts.py`, `test_net_gross_identity.py` | net/gross exposure and mean-turnover identities; export digest stability |
| `metrics/overfitting` | `test_backtest_overfitting.py`, `test_reality_identities.py` | DSR ≤ PSR at a non-negative hurdle; both in [0,1]; PSR/CSCV/DSR ledger identities |
| `metrics/cross_section` | `test_cross_sectional_rankic.py` | Spearman invariant to within-date asset permutation; sign-flip anti-symmetry; shuffled labels destroy a planted edge; null does not over-reject |

The other 54 modules have **no generated-input coverage**, including every
module named in the honesty contract's first rule:

| Priority | Module | Public estimators | Existing coverage |
|---|---|---|---|
| P0 | `metrics/scoring` | `pinball_loss`, `mean_pinball`, `coverage`, `interval_width`, `quantile_crossing_rate`, `rearrange_quantiles`, `crps_from_quantiles`, `crps_gaussian`, `crps_student_t`, `crps_gaussian_mixture`, `gaussian_mixture_quantiles`, `crps_empirical`, `log_score_gaussian`, `qlike`, `overlap_aware_qlike`, `name_level_qlike`, `one_step_density_summary`, `pearson_ic`, `rank_ic`, `icir`, `pit_values`, `fissler_ziegel_loss` | Known-answer: `tests/unit/metrics/test_stats_audit.py`, `tests/unit/research/{test_scoring_edges,test_crps_closed_form,test_fissler_ziegel,test_overlap_aware_scoring}.py`, `tests/regression/test_scoring_frozen.py`. Frozen pins: 2 values |
| P0 | `metrics/probability` | `brier_score`, `log_loss`, `expected_calibration_error`, `kupiec_pof`, `pit_ks`, `christoffersen_independence`, `christoffersen_cc`, `acerbi_szekely_z1/z2` | Known-answer + edges: `tests/unit/core/test_probability.py` |
| P0 | `metrics/risk` | `historical_var`, `historical_es`, `gaussian_var`, `gaussian_es`, `var_es_from_return_quantiles`, `losses_from_returns` | Example-based under `tests/unit/risk/` (12 files import `metrics`) |
| P1 | `metrics/conformal` | `conformal_quantile`, `cqr_scores`, `onesided_scores`, `expand_interval`, `covered`, `set_metrics`, `conditional_coverage`, `worst_slice_coverage` | `tests/unit/research/test_numeric_return_contracts.py` (shape/dtype/non-mutation contracts, example-based) |
| P1 | `metrics/evalues`, `metrics/anytime_fdr`, `metrics/e_detectors` | `e_process`, `e_process_threshold`, `e_process_loss_diff`, `e_process_dm`, `e_bh`, `stopped_e_bh`, `ELond` | `tests/unit/core/test_anytime_fdr.py` — **seeded Monte-Carlo FDR-under-global-null** checks (300 reps). Excellent statistics, but every draw is one fixed seed, so a threshold off-by-one or a boundary `alpha` would not be found |
| P1 | `metrics/energy_score` | `energy_score`, `threshold_energy_score`, `energy_score_curve` | `tests/unit/core/test_energy_score.py` |
| P1 | `metrics/density_forecast` | `pit_histogram`, `berkowitz_test`, `pit_autocorrelation` | `tests/unit/research/test_density_forecast.py` |
| P1 | `metrics/var_backtest`, `metrics/es_backtest` | `kupiec_test`, `christoffersen_test`, `tuff_test`, `basel_zone`, `acerbi_szekely_test`, `mcneil_frey_test`, `exceedance_residuals`, `du_escanciano_test` | `tests/unit/risk/test_{var,es}_backtest.py` |
| P2 | `metrics/snooping` | `reality_check`, `spa_test`, `stepm`, `model_confidence_set` | `test_stats_audit.py` (size ≈ nominal under global null) + `test_reality_*` |
| P2 | `metrics/inference`, `metrics/hac`, `metrics/bootstrap` | NW/QS/AM HAC, DM, stationary bootstrap, Politis–White block length, wild/pairs/sieve bootstrap | `test_stats_audit.py` cross-checks `optimal_block_length` against `arch` |
| P2 | `metrics/serial`, `metrics/fractal`, `metrics/entropy`, `metrics/extremes`, `metrics/spectral_risk`, `metrics/entropic_risk`, `metrics/twosample`, `metrics/concordance`, `metrics/calibration_tests`, `metrics/robust_location`, `metrics/scale_tests`, `metrics/distribution`, `metrics/kde`, `metrics/dependence`, `metrics/mic`, `metrics/dcca` | Large canonical batteries (Lo–MacKinlay VR, Chow–Denning, Wright, runs, Ljung–Box, ARCH-LM, JB, ADF/KPSS/PP/ZA; Hill/Pickands/Dekkers, GPD POT, GEV; Shannon/ApEn/SampEn/permutation/LZ/spectral/transfer/dispersion entropy; Hurst R/S, DFA, Katz/Higuchi/Sevcik; KS/energy/permutation/CvM; Spiegelhalter z; …) | Known-answer tests across `tests/unit/{core,metrics,research,risk,models}` |
| P2 | `metrics/purged_cv`, `metrics/overfitting` (partly done), `metrics/research_evaluation`, `metrics/forecast_accuracy`, `metrics/forecast_eval`, `metrics/perf_ratios`, `metrics/drawdown`, `metrics/panel`, `metrics/regression`, `metrics/vol_eval`, `metrics/path_rankic`, `metrics/feature_select`, `metrics/agreement`, `metrics/cluster_validity`, `metrics/empirical_likelihood`, `metrics/ledoit_wolf_sharpe`, `metrics/liquidity_extra`, `metrics/utility`, `metrics/direction`, `metrics/calibration2`, `metrics/conformal_martingale`, `metrics/watch`, `metrics/spectral_risk` | — | Example-based |

Adjacent-but-related: `utils/numeric` (`require_finite`, `clip_positive`,
`positive_integral_count`, `require_upper_tail_alpha`, `numeric_hessian`) has
contract tests in `test_numeric_return_contracts.py` but no property coverage,
despite being the shared guard layer for the whole mypy numeric-return
ratchet. `models/covariance` does have a `@given` test
(`tests/unit/models/test_covariance.py`) — the only `models/` module that does.

Why this matters more here than in a typical codebase: the honesty contract
makes proper scores the *only* admissible research evidence, and receipts are
the immutable record of that evidence. An estimator bug in `metrics/scoring`
therefore does not just produce a wrong number — it produces a wrong number
inside an immutable, hash-sealed artifact that downstream promotion logic
trusts. That is the definition of a high-leverage test target.

### 4.8 Tolerance and determinism practice

- `pytest.approx` in 260 files; `np.testing.assert_allclose` in 37; ULP-based
  assertions in **0**. Tolerances are almost always ad-hoc literals
  (`rel=1e-9, abs=1e-9`, `rtol=1e-12, atol=1e-12`) with no recorded rationale.
- `deadline=None` in 27 files, `max_examples` in 35, `derandomize=True` in 6,
  `suppress_health_check` in 7, `print_blob=True` in **1**. Zero
  `@example(...)` regression pins and zero `# Reproduce failure:` comments —
  so counterexamples found in CI are not being committed as permanent examples.
  With `database=None` in the adversarial profiles and the `.hypothesis` dir
  gitignored, a nightly-only failure at 80 examples is reproducible *only* if
  someone reads the log and re-pins it by hand.
- `hypothesis.extra.numpy`: **0** files.
- Determinism: good discipline — seeded `np.random.default_rng`, derandomized
  CI profiles, byte-identity for the fast path, `set_global_seed` property
  tested. The `Flaky` risk from wall-clock/unseeded randomness is handled by
  convention rather than by a lint rule.

---

## 5. Gap summary

| # | Gap | Severity | Effort |
|---|---|---|---|
| G1 | Working-tree `pyproject.toml`/`uv.lock` revert breaks `--strict-markers`, the coverage ratchet, `testpaths` lane separation, and three CI lanes | **Blocking** | Small (restore or land deliberately) |
| G2 | 54/58 `metrics/` modules have no generated-input coverage; all honesty-contract scoring modules are in that set | High | Large, incremental |
| G3 | No shared strategy library for the property lane; `hypothesis.extra.numpy` unused | Medium | Small |
| G4 | Mutation testing not in CI; no floors, no ratchet, 3/733 modules | Medium | Medium |
| G5 | Nightly Hypothesis budget applies to 6 of 30 property files | Medium | Small |
| G6 | No ULP-level or backward-error assertions; tolerance literals undocumented | Medium | Medium |
| G7 | No counterexample pinning discipline (`@example`, `Reproduce failure` blobs) | Medium | Small (process) |
| G8 | No coverage-guided fuzzing of parser boundaries (`sources/normalize`, PIT gates, canonical JSON) | Medium | Medium |
| G9 | Data-engine metamorphic relations are example-based (corporate-action wealth conservation is a single hand-built 2-bar fixture) | Medium | Small |
| G10 | `synthetic` marker applied to 22 files where it is true of nearly all | Low | Small |
| G11 | `tests/tests/` (233 untracked files) and `.backup-prerestore/` mirror trees | Low | Small |
| G12 | Golden masters for receipt/notebook/scorecard shape live in CI YAML heredocs, untestable locally | Low | Medium |

---

## 6. Adoption plan

Sequenced so each phase lands independently and stays green. Nothing here
requires a new dependency except Phase 4 (Atheris) and the optional
pytest-regtest/syrupy in Phase 5.

### 6.0 Phase 0 — unblock (must come first)

1. Resolve G1: restore HEAD's `pyproject.toml` + `uv.lock`, or land the
   intended rename in one commit with `uv lock` regenerated and the six
   markers, 80% floor, proofcore floors, `testpaths`, extras, scripts, mccabe
   ceiling and mypy packages preserved. Verify with
   `uv sync --frozen --all-groups --all-extras` and
   `uv run pytest --collect-only -q tests/property tests/native tests/perf`.
2. Resolve G11: delete or relocate `tests/tests/` and `.backup-prerestore/`
   (per AGENTS.md the mirror was dropped; a stray must stay out of collection).
3. Add `.hypothesis/` to the repo `.gitignore` explicitly. Hypothesis
   self-generates `.hypothesis/.gitignore`, which works, but an explicit entry
   is the documented house convention and survives a `.hypothesis` wipe.

### 6.1 Phase 1 — shared strategy library + invariant catalog scaffolding

New `tests/property/_numeric_strategies.py` (mirrors `_profiles.py`, no new
dependency; uses `hypothesis.extra.numpy`, which ships with Hypothesis):

```python
"""Shared numeric strategies for the property lane.

Domain knowledge lives here so it cannot drift between files. Every strategy
is finite by default; the non-finite variants are opt-in because the harness
convention is that estimators mask non-finite input honestly rather than
coerce it (see metrics/scoring.py, metrics/probability.py).
"""

from hypothesis import strategies as st
from hypothesis.extra import numpy as hynp
import numpy as np

# Returns: plausible per-bar simple returns. Wide enough to hit -90% and 10x.
ret_element = st.floats(min_value=-0.9, max_value=10.0, allow_nan=False, allow_infinity=False)
returns_1d = hynp.arrays(np.float64, hynp.array_shapes(min_dims=1, max_dims=1,
                         min_side=1, max_side=64), elements=ret_element)

# Strictly positive scale quantities (variances, sigmas, prices, realized vol).
pos_element = st.floats(min_value=1e-8, max_value=1e8, allow_nan=False, allow_infinity=False)
pos_1d = hynp.arrays(np.float64, hynp.array_shapes(min_dims=1, max_dims=1,
                     min_side=1, max_side=64), elements=pos_element)

# Quantile levels and probabilities, exclusive of the endpoints the estimators reject.
tau = st.floats(min_value=1e-3, max_value=1 - 1e-3, allow_nan=False, allow_infinity=False)
alpha = st.floats(min_value=1e-3, max_value=1 - 1e-3, allow_nan=False, allow_infinity=False)
sorted_taus = st.lists(tau, min_size=2, max_size=23, unique=True).map(sorted)

# Observations: arbitrary finite reals, symmetric around zero.
obs_1d = hynp.arrays(np.float64, hynp.array_shapes(min_dims=1, max_dims=1,
                     min_side=1, max_side=64),
                     elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))

# Non-finite frontier: for the fail-closed half of every contract.
nonfinite = st.sampled_from([float("nan"), float("inf"), float("-inf")])

def with_one_nonfinite(x: np.ndarray, i: int) -> np.ndarray:
    out = x.copy()
    out[i % max(out.size, 1)] = float("nan")
    return out
```

Plus a `tests/property/conftest.py` exposing these as fixtures and applying
`synthetic` automatically:

```python
def pytest_collection_modifyitems(config, items):
    """Every property test is SYNTHETIC by construction (honesty contract §2)."""
    for item in items:
        if item.nodeid.startswith("tests/property/") or item.nodeid.startswith("tests/formal/"):
            item.add_marker(pytest.mark.synthetic)
```

And a catalog document convention: each metric module gets an
`INVARIANTS` block in its module docstring (or a sibling
`docs/MATH_SPEC.md` section) listing the properties that must hold, each with
its citation. The test file then has one `@given` per listed invariant and a
comment naming it. This makes "is the catalog complete?" a reviewable question
instead of an implicit one, and it is the artifact a mutation run can be
checked against.

### 6.2 Phase 2 — invariant catalogs per metric module

Below are concrete, citable catalogs for the P0/P1 modules. Each row is one
`@given` test. "Source" is the property's justification, not a citation of
this repo's code.

#### 6.2.1 `metrics/scoring.py` — proper scores and quantile machinery

Proper scoring rules: [Gneiting & Raftery, *Strictly Proper Scoring Rules,
Prediction, and Estimation*, JASA 102(477):359–378, 2007]. Quantile
consistency: [Ehm, Gneiting, Jordan & Krüger, JRSS-B 78(3):505–562, 2016].
Joint (VaR, ES) elicitability: [Fissler & Ziegel, 2016]. Interval/quantile
evaluation practice: [Bracher, Ray, Gneiting & Reich, *Evaluating epidemic
forecasts in an interval format*, PLOS Comput. Biol. 17(2):e1008618, 2021].

| # | Invariant | Shape |
|---|---|---|
| S1 | **Non-negativity.** `pinball_loss ≥ 0` elementwise for all finite `y, q, τ∈(0,1)`; `mean_pinball ≥ 0` or honest NaN | `@given(y=obs_1d, q=obs_1d, τ=tau)` |
| S2 | **Zero at the truth.** `pinball_loss(y, y, τ) == 0`; `crps_gaussian(y, y, σ→0) → 0`; `qlike(v, v) == 0` | `@given(y=obs_1d, τ=tau)` |
| S3 | **Quantile-crossing consistency.** For a *crossed* quantile vector, `quantile_crossing_rate > 0`, and `rearrange_quantiles` output is monotone non-decreasing with `crossing_rate == 0` | `@given(q=obs_1d, taus=sorted_taus)` |
| S4 | **Rearrangement idempotence.** `rearrange(rearrange(q)) == rearrange(q)` exactly (a sort is idempotent — assert bit equality, no tolerance) | `@given(q=obs_1d)` |
| S5 | **Rearrangement is measure-preserving.** `sorted(rearrange(q)) == sorted(q)`; mean and multiset unchanged | `@given(q=obs_1d)` |
| S6 | **CRPS ≥ 0 and CRPS-from-quantiles → closed form.** `crps_gaussian ≥ 0`; `crps_from_quantiles` over a dense τ grid converges to `2 × crps_gaussian` within a documented Riemann bound that shrinks as the grid refines | `@given(y=obs_1d, mu=obs_1d, sigma=pos_1d)`; convergence test over grid sizes 200/2000 |
| S7 | **Scale homogeneity (the CRPS analogue of S12).** `crps_gaussian(a·y, a·μ, a·σ) == a·crps_gaussian(y, μ, σ)` for `a>0`; same for `crps_student_t` with `ν` fixed | `@given(y, mu, sigma, a=pos_element)` |
| S8 | **Translation equivariance.** `crps_gaussian(y+c, μ+c, σ) == crps_gaussian(y, μ, σ)`; `pinball_loss(y+c, q+c, τ) == pinball_loss(y, q, τ)` | `@given(y, mu, sigma, c=obs element)` |
| S9 | **Student-t → Gaussian limit.** `crps_student_t(y, μ, σ, ν) → crps_gaussian(y, μ, σ')` as `ν → ∞` with the correct scale map `σ' = σ·√((ν−2)/ν)`; monotone approach asserted, not just the endpoint | `@given(y, mu, sigma, ν=st.floats(3, 1e6))` |
| S10 | **Mixture consistency.** A one-component mixture equals the single Gaussian; a two-component mixture with weights `(1,0)` collapses to component 1; `gaussian_mixture_quantiles` of a single component equals `norm.ppf` | `@given(...)` |
| S11 | **Empirical CRPS ↔ ensemble identity.** `crps_empirical(y, sample)` equals the energy-distance form `(1/n)Σ|xᵢ−y| − (1/(2n(n−1)))Σ|xᵢ−xⱼ|` to `1e-12`; and `crps_empirical(y, [y]) == 0` | `@given(y, sample=pos/obs_1d)` |
| S12 | **QLIKE positivity and minimizer.** `qlike ≥ 0` for all valid `(y≥0, ŷ>0)`; minimized at `ŷ = y`; strictly convex in `ŷ` (second difference > 0); negative/non-finite entries masked, never clipped into a finite score | `@given(y=pos_1d, yhat=pos_1d)` |
| S13 | **QLIKE asymmetry (the reason QLIKE is used for variance).** Under-prediction is penalized more than equal-magnitude over-prediction: `qlike(v, v/k) > qlike(v, v·k)` for `k>1` | `@given(v=pos_element, k=pos>1)` |
| S14 | **Coverage bounds and monotonicity.** `0 ≤ coverage ≤ 1` or honest NaN; widening `[lower, upper]` pointwise cannot decrease coverage; `coverage` of an all-non-finite input is NaN | `@given(y, lower, upper)` with a monotone perturbation |
| S15 | **Interval width non-negativity and additivity.** `interval_width ≥ 0`; width of a union of nested intervals equals the widest | `@given(lower, upper)` |
| S16 | **PIT in-range and uniformity.** `pit_values ∈ [0,1]`; under a correctly-specified Gaussian forecast the PIT sample passes `pit_ks` at a nominal rate over many seeds; τ-crossing in the input cannot push PIT out of range | `@given(y, q, taus)`; seeded MC for uniformity |
| S17 | **IC bounds.** `|pearson_ic| ≤ 1`, `|rank_ic| ≤ 1` or honest NaN; `rank_ic` is invariant to any strictly-monotone transform of either argument; `ic(−x, y) == −ic(x, y)`; `ic(x, x) == 1` | `@given(pred, realized)` |
| S18 | **ICIR scale behaviour.** `icir` is invariant to a positive rescale of the IC series and sign-flips under negation | `@given(ics, scale)` |
| S19 | **Fissler–Ziegel joint propriety.** `fissler_ziegel_loss ≥` its value at the true `(VaR, ES)` in expectation; ES is not elicitable alone, so the pair must be tested jointly (this generalizes the existing 400k-draw example into `@given(alpha, tail_mass, n)`) | `@given(α=alpha, p=tail prob, n=int)` |
| S20 | **Overlap-aware masking is causal and count-correct.** `nonoverlapping_origin_mask` selects ≤ `ceil(n/h)` origins, never two within `h−1` bars, and always keeps the first; `_per_name_nonoverlapping_mask` per name equals the mask computed on that name alone | `@given(positions, h=int≥1)` |
| S21 | **Fail-closed on shape.** Any length mismatch raises `ValueError`; any τ/α outside its documented open interval raises; empty input returns honest NaN (never `0.0` — a zero score would look like a perfect forecast) | `@given(...)` |
| S22 | **Non-mutation.** No estimator mutates its input arrays (the repo already asserts this for `clip_positive`/`cqr_scores` in `test_numeric_return_contracts.py`; generalize) | `@given(...)` |

#### 6.2.2 `metrics/probability.py` — Brier, log-loss, ECE, hit tests

| # | Invariant |
|---|---|
| P1 | `0 ≤ brier_score ≤ 1` or honest NaN; `brier(p, p) == 0`; Brier of the true frequency equals the empirical variance of the indicator |
| P2 | **Brier ≤ naive constant.** For any `p`, `brier(p, y) ≤ max(brier(0,y), brier(1,y))`; and the constant forecast `p̄ = mean(y)` minimizes Brier over constants (verify by grid) |
| P3 | Out-of-range probabilities are **masked, not clipped** — `brier_score([1.5], [1])` is NaN, not `0.25`. This is a documented honesty choice and deserves a generated test, because a silent clip would fabricate a good-looking score |
| P4 | `log_loss ≥ 0`; `log_loss` strictly increases as a confident-and-correct forecast moves toward the wrong endpoint; `log_loss` dominates Brier ordering on the same data only where theory says it does (assert the *known* non-comparability rather than an ordering) |
| P5 | `0 ≤ expected_calibration_error ≤ 1`; a perfectly calibrated input (`p == y` per bin) gives ECE 0; **ECE is bin-dependent** — assert monotone non-increase as `n_bins` grows toward one-bin-per-observation on a calibrated sample, and assert the documented NaN for `n_bins < 1` or `p.size < n_bins` |
| P6 | **Kupiec size.** Under a true hit rate equal to `α`, `kupiec_pof` p-values are Uniform(0,1) over seeds; LR → 0 as the empirical rate → α; `n < 10` → NaN triple; `α ∉ (0,1)` → NaN triple; all-hit and all-miss → NaN LR (documented) |
| P7 | **Kupiec LR non-negativity and monotonicity.** `lr ≥ 0`; `lr` strictly increases as the hit rate departs from `α` in either direction (two-sided) |
| P8 | **PIT KS.** Under Uniform(0,1) input the p-value is uniform over seeds; under mass piled at 0/1 it rejects; `n < 8` → NaN pair; input is clipped to `(0,1)` internally so exact 0/1 must not produce a NaN statistic |
| P9 | **Christoffersen independence.** A serially-independent hit sequence at rate `α` yields a non-rejecting independence p-value at nominal rate over seeds; an alternating/clumped sequence rejects; LR ≥ 0 |
| P10 | **Christoffersen CC = POF + independence.** The conditional-coverage LR equals the sum of the two component LRs to `1e-10` — an algebraic identity that must hold for *all* inputs, and is currently untested as such |
| P11 | **Acerbi–Székely z1/z2.** Under a correctly specified VaR the statistics are approximately standard normal over seeds; z-statistics are monotone in the magnitude of tail miss; `α ∉ (0,1)` → `ValueError` (note this module *raises* where others return NaN — the asymmetry is deliberate and must be pinned) |

#### 6.2.3 `metrics/risk.py`, `metrics/conformal.py`

| # | Invariant |
|---|---|
| R1 | `losses_from_returns(r) == −r` exactly; empty → empty |
| R2 | **VaR monotone in α.** `historical_var(x, α₁) ≤ historical_var(x, α₂)` for `α₁ < α₂`; `gaussian_var` likewise |
| R3 | **ES ≥ VaR** at the same α (for losses; sign convention must be pinned, since a sign slip here is silent and catastrophic) |
| R4 | **ES monotone in α** and **translation-equivariant**: `ES(x + c) == ES(x) + c`; **positively homogeneous**: `ES(a·x) == a·ES(x)` for `a > 0` |
| R5 | **Empirical tail-mass consistency.** `historical_es` equals the mean of losses above the empirical VaR including fractional mass — assert against a direct recomputation on generated samples, including the non-integer-tail-mass case the docstring calls out |
| R6 | **Conformal coverage guarantee.** For exchangeable calibration/test data, the empirical coverage of `expand_interval(conformal_quantile(scores, α))` is ≥ `1 − α` across many seeds; the finite-sample quantile `k = ceil((n+1)(1−α))` clipped to `[1, n]` must be asserted directly, including the `k > n` clip branch which the docstring flags as un-coverable at that α |
| R7 | **CQR score identity.** `cqr_scores(y, lo, hi)` equals `max(lo − y, y − hi)`; with `scale`, the normalized form; `scale == 0` must not produce `inf` silently (the existing example asserts `−1e12` — pin that as the documented sentinel and assert it is finite and below any real score) |
| R8 | `set_metrics` / `covered` return values in `[0,1]`; `worst_slice_coverage ≤ conditional_coverage ≤ 1`; `assign_terciles` partitions every index exactly once |
| R9 | **No-Sharpe guard.** No function in `metrics/risk` or `metrics/conformal` returns a key in `FORBIDDEN_RESEARCH_METRIC_KEYS` (a structural test that walks return dicts) |

#### 6.2.4 `metrics/evalues.py`, `metrics/anytime_fdr.py`, `metrics/e_detectors.py`

| # | Invariant |
|---|---|
| E1 | **Ville's inequality, deterministically.** For an e-process under the null, `P(sup_t E_t ≥ 1/α) ≤ α`. The existing tests verify this by seeded Monte-Carlo at fixed parameters; the property version generates `α`, path length, sharing structure and alternative, and asserts the *empirical* rate over reps stays under `α + tolerance` — with the tolerance derived from a Hoeffding bound on `reps`, not chosen by eye |
| E2 | **E-value non-negativity and unit expectation.** `e_process` steps are ≥ 0; `E_H0[e_step] == 1` to `1e-12` for the Bernoulli betting e-variable, computed in closed form (this is a pure algebra check and needs no randomness) |
| E3 | **Monotonicity in the evidence.** The e-process is non-decreasing in the number of favourable observations; `e_process_threshold` is monotone in its threshold |
| E4 | **e-BH FDR control under arbitrary dependence.** e-BH controls FDR at α under *any* dependence — so generate maximally-dependent e-values (all rows sharing the same underlying flips, the `n_shared` knob already in the test helper) and assert control. This is the property that distinguishes e-BH from p-value BH and it is the one most worth generating |
| E5 | **e-BH monotonicity.** Increasing any e-value cannot reduce the rejection count; `stopped_e_bh` rejects at least as much as `e_bh` at the same α on the same terminal values |
| E6 | **Optional stopping.** `stopped_e_bh` at an adversarially-chosen stopping time (generated `tau`, including data-dependent stops) still controls FDR — the entire point of the stopped variant |
| E7 | **e-LOND γ-sequence contract.** Rejections are monotone in the wealth; the default and overridden γ sequences both control FDR; `gamma` must sum appropriately for the guarantee to hold — assert the documented precondition is enforced, not assumed |
| E8 | **Fail-closed.** Non-finite or negative e-values raise or are masked per the documented contract; `α ∉ (0,1)` is rejected |
| E9 | **E-detector ARL.** Under the null, the average run length of an e-detector alarm is ≥ `1/α` over seeds; under a generated mean-shift, detection time decreases with shift size (monotone power) |

#### 6.2.5 `metrics/energy_score.py`, `metrics/density_forecast.py`, `metrics/twosample.py`

| # | Invariant |
|---|---|
| N1 | **Energy score non-negativity.** `energy_score ≥ 0` for all ensembles/observations; `== 0` iff the ensemble is degenerate at the observation (assert both directions on generated inputs) |
| N2 | **Permutation invariance in the ensemble.** Reordering ensemble members leaves the score bit-identical (the score is a symmetric U-statistic — assert exact equality, since a symmetric sum reordered should agree to within the documented `rtol`, and *record* the achieved deviation) |
| N3 | **Dimension reduction.** In `d = 1` the energy score equals the CRPS up to the documented factor — a duality check between two independently-implemented proper scores |
| N4 | **Threshold weighting degenerates correctly.** `threshold_energy_score(weight=1.0) == energy_score` exactly; larger `weight` weakly increases the score for tail errors and leaves central-error cases unchanged; `weight < 1` must be rejected per the documented `weight >= 1` contract |
| N5 | **Energy score curve monotonicity.** `energy_score_curve` is non-decreasing in the threshold |
| N6 | **PIT histogram normalization.** Bin counts sum to `n`; bins are within `[0,1]`; a uniform sample yields near-flat counts over seeds |
| N7 | **Berkowitz statistic non-negativity** and asymptotic χ²(3) behaviour under correct specification over seeds |
| N8 | **Two-sample tests.** `energy_distance(x, x) == 0`; symmetric in its arguments; `ks_two_sample` statistic in `[0,1]`; `permutation_test` p-value in `[0,1]` and invariant to the permutation seed once `reps` is large enough; identical samples never reject |

#### 6.2.6 `metrics/serial.py`, `metrics/fractal.py`, `metrics/entropy.py`, `metrics/hac.py`, `metrics/bootstrap.py`

| # | Invariant |
|---|---|
| T1 | **White noise ⇒ no autocorrelation.** On generated iid Gaussian input, `ljung_box` / `box_pierce` p-values are uniform over seeds; `autocorrelation(x, 0) == 1` exactly |
| T2 | **ARCH-LM power.** On generated GARCH(1,1) input the LM test rejects at a rate well above nominal; on iid it does not |
| T3 | **Variance-ratio bounds and sign.** `variance_ratio_test` statistic → 1 for a random walk over long horizons; Wright's rank/sign variants agree in sign with the Lo–MacKinlay variant on the same generated path; Chow–Denning multiple statistic ≥ each individual one |
| T4 | **Hurst bounds.** `hurst_rs ∈ [0, 1]` for generated paths; a constructed trending path gives `H > 0.5`, an anti-persistent one `H < 0.5`; `dfa_hurst` agrees in sign of deviation from 0.5 |
| T5 | **Fractal-dimension ordering and bounds.** Katz/Higuchi/Sevcik FDs lie in their documented ranges; a smooth path scores lower FD than white noise; all three agree in *ordering* on the same generated pair of paths (a metamorphic relation across three independent estimators of the same quantity) |
| T6 | **Entropy bounds.** Shannon entropy of a uniform `k`-symbol source equals `log k`; permutation entropy ∈ [0,1] and equals 1 on iid noise, → 0 on a monotone path; approximate/sample entropy decrease with regularity; spectral entropy ∈ [0,1]; transfer entropy ≥ 0 and ≈ 0 for independent generated series |
| T7 | **HAC ≥ iid variance.** For a positively autocorrelated generated process, `newey_west_lrv ≥` the iid variance estimate; kernel weights are non-negative and symmetric; `andrews_bandwidth` is a positive integer within its documented clamp |
| T8 | **Bootstrap distributional fidelity.** Bootstrap CIs from `wild_bootstrap_residuals`, `rademacher_bootstrap`, `pairs_bootstrap`, `sieve_bootstrap_residuals` cover the true mean at ≥ nominal rate on generated data with known mean; `circular_block_indices` returns contiguous wrapped blocks of exactly the requested length, covering the sample uniformly over seeds |
| T9 | **Block-length selection cross-check.** `optimal_block_length` agrees with `arch.bootstrap.optimal_block_length` within a documented tolerance across generated AR(1) processes with varying `φ` (the existing known-answer test does this at one `φ`) |
| T10 | **Purged CV correctness.** `purged_kfold_indices` never places a train index within the purge window of a test index, never places a test index inside the embargo of the *next* fold, and covers every index exactly once across folds — for generated `n`, `k`, horizon `h`, embargo `e`, including the degenerate `h ≥ n/k` case that must fail closed |

#### 6.2.7 `metrics/snooping.py`, `metrics/overfitting.py` (extend existing)

| # | Invariant |
|---|---|
| D1 | **Size control under the global null.** `reality_check`, `spa_test` (all three variants), `stepm` reject at ≈ α over generated null datasets; the tolerance comes from a binomial/Hoeffding bound on `reps`, not a magic number |
| D2 | **StepM step-down monotonicity.** Adjusted p-values are non-decreasing along the step-down order; StepM rejects a superset of what single-test Bonferroni rejects |
| D3 | **MCS containment.** Under the global null `model_confidence_set` retains all models with probability ≥ 1 − α; with a planted inferior model it is eliminated; the elimination sequence is monotone (once out, never back in) |
| D4 | **SPA studentization ⇒ scale invariance.** Rescaling every model's loss series by a common positive constant leaves the SPA p-value unchanged (already asserted at one configuration; generate it) |
| D5 | **PSR/DSR ordering.** `deflated_sharpe ≤ probabilistic_sharpe` at a non-negative hurdle (already a property); extend: PSR is monotone increasing in the observed statistic, monotone decreasing in `n_trials`, and both lie in `[0,1]` |
| D6 | **PBO symmetry.** On generated performance matrices with no planted edge, `probability_of_backtest_overfitting ≈ 0.5`; with a planted edge it departs from 0.5 in the documented direction |
| D7 | **MinTRL monotonicity.** `min_track_record_length` increases with the required confidence and with the Sharpe hurdle |

#### 6.2.8 `utils/numeric.py` (guard layer — small, high leverage)

| # | Invariant |
|---|---|
| U1 | `clip_positive` never returns a value below `floor`, never returns non-finite, is idempotent, and does not mutate its input; scalar in → `float` out, array in → `float64` array out with **shape and dtype preserved for `()`, `(0,)`, `(n,)`, `(n,m)`** (the existing parametrized test covers four shapes at one value; generate the values) |
| U2 | `require_finite` raises iff any element is non-finite; returns the identical object otherwise |
| U3 | `positive_integral_count` returns 0 (fail closed) for bools, strings, non-integers, non-finite, and non-positive values; returns the exact int otherwise; **never coerces** — assert `True → 0`, `2.0 → 2`, `2.5 → 0`, `nan → 0`, `inf → 0`, `−3 → 0` |
| U4 | `require_upper_tail_alpha` accepts exactly `(0.5, 1)` and rejects everything else including the endpoints |
| U5 | `numeric_hessian` is symmetric for any generated objective; equals the analytic Hessian of a generated quadratic form within a tolerance derived from `HESSIAN_STEP_SCALE` and the quadratic's conditioning; degrades gracefully (finite, or documented failure) as conditioning worsens |

### 6.3 Phase 3 — mutation-testing gate

**Design principle:** follow the Google pattern (diff-scoped, precision over
recall, review-surfaced) rather than a repo-wide percentage. Extend the
existing, already-correct `scripts/mutation_score.py` rather than replacing it.

**Protected module set (tier 1, blocking).** Chosen by the honesty contract and
by blast radius, not by size:

| Module | Rationale | Current score |
|---|---|---|
| `execution/costs.py` | Money path | 100.0% (baseline) |
| `metrics/returns.py` | Money path | 89.3% (baseline) |
| `utils/hashing.py` | Receipt immutability | 86.6% (baseline) |
| `metrics/scoring.py` | Honesty-contract evidence | **not yet measured** |
| `metrics/probability.py` | Honesty-contract evidence | not yet measured |
| `research/verify.py` | Receipt verification (935 lines) | not yet measured |
| `research/catalog/*` (`predicates`, `consistency`, `never_equate`, `receipt`) | Forbidden-key enforcement | not yet measured |
| `data/point_in_time.py` + `pit/` | Look-ahead prevention | not yet measured |
| `data/sources/normalize.py` | Fail-closed on untrusted input | not yet measured |
| `paper/ledger.py` | Ledger schema validity | not yet measured |

`docs/SOTA_CANON_ROADMAP_2026_09.md` §2.4 item (6) already names
`research/verify.py` + `catalog.py` as the mutation-testing target; this list
extends it with the scoring and PIT layers.

**Gate mechanics.**

1. **Baseline file, not a threshold.** Keep
   `tests/property/mutation_scores.json` as the committed record and add a
   `tests/unit/test_mutation_baseline.py` that asserts:
   - every tier-1 module has a recorded score (a *missing* entry fails — this
     is what stops the record going stale);
   - `score_percent ≥ baseline_score_percent` per module (ratchet, mirroring
     `test_quality_ratchet.py`'s existing style: "raise, never lower");
   - `survived ≤ baseline_survived` per module, **and every survivor is listed
     in `equivalent_survivors` with a non-empty justification string**. The
     current file already does this by convention; make it a check.
   - `mutmut_version` and `numpy` version recorded, so a score change caused by
     a tool upgrade is visible rather than mysterious.
2. **Framing.** Gate on "no new surviving mutants in protected modules", not on
   "score above X%". The practitioner literature is unanimous that a
   percentage target produces over-specified tests; a survivor-count ratchet
   produces targeted ones.
3. **CI placement.** A **scheduled weekly job** (not per-PR) that runs
   `scripts/mutation_score.py` over the tier-1 set and fails on regression
   against the committed baseline. Measured cost is ~55 s per ~200-mutant
   module at `HYPOTHESIS_PROFILE=ci`, so ten modules is roughly 10–20 minutes
   serially, less with `--max-children 4`. Add a **manual `workflow_dispatch`**
   trigger for pre-release runs.
4. **PR-scoped diagnostic (optional, later).** A diff-scoped job that mutates
   only files touched by the PR and *comments* survivors without failing. This
   is the Google model and it is what makes mutation testing survive long-term.
   Do not make it blocking until the weekly job has been stable for a few
   weeks (the "diagnostic first, gate later" advice in every practitioner
   write-up).
5. **Tool choice.** Stay on mutmut 3 — `scripts/mutation_score.py` already
   depends on its v3 trampoline for test-impact selection, which is the single
   biggest cost lever, and its `export-cicd-stats` gives machine-readable
   counts. Consider Cosmic Ray only if the tier-1 set grows past ~25 modules
   and the distributor model becomes worth the extra dependency.
6. **Equivalent-mutant discipline.** Keep requiring a written argument per
   survivor cluster, as the current file does. Where an equivalence class is
   large (the 19-mutant `json.dumps` sort-key cluster in `hashing.py`), the
   correct response is *not* more mutants — it is a targeted property test on
   canonicalization (byte-level ordering stability across dict/list/str/nested
   combinations, non-finite rejection, unicode normalization), which is
   §6.2.8-adjacent work.

### 6.4 Phase 4 — data-engine metamorphic relations and parser fuzzing

**Metamorphic relations to add** (each becomes one `@given` test; the
conservation ones are the highest value because they are the relations a
silent data bug violates):

| # | Relation | Target |
|---|---|---|
| M1 | **Corporate-action wealth conservation, generated.** Split + cash dividend on the same ex-date conserves wealth for generated `(factor, amount, raw_close)` triples; total-return index stays on the split-adjusted base. The existing `tests/regression/test_split_dividend_conservation.py` pins one hand-built 2-bar case with a documented counterexample ("dividing the dividend by the raw previous close (100) instead of the split-adjusted previous close (50) reports 49.5") — that reasoning is exactly a metamorphic relation and should be generated over many parameter triples | `data/corporate_actions::adjust_prices`, `cumulative_split_factors`, `attach_dividends` |
| M2 | **Adjustment idempotence and order independence.** Applying a split twice with factor `k` equals once with `k²`; applying actions sorted by `event_time` in any stable order gives the same result; a zero-action table is the identity | `adjust_prices` |
| M3 | **Resample conservation (extend existing).** Volume, and the OHLC envelope (`min(low)`, `max(high)`, first `open`, last `close`) are conserved under resampling for generated bar counts, generated bar sizes, and generated missing-bar patterns; resampling to a coarser then a finer grid does not invent bars | `data/adapters/hf_ohlcv_1m::resample_ohlcv` |
| M4 | **As-of causality (extend existing).** For any generated append/restate sequence, `asof(t)` never returns a row with `known_at > t`; a restatement published after `t` leaves `asof(t)` **byte-identical**; ties on `known_at` resolve to latest arrival, matching the pure-Python model | `pit::PitVault`, `data/point_in_time` |
| M5 | **Session tiling (extend existing).** Bars exactly partition `[open_utc, close_utc)` with no gaps or overlaps for every generated (calendar, bar size, window) triple across DST transitions and half-days; every in-session timestamp maps to exactly one slot; `assign_bars` is a total function on generated timestamps | `calendars` |
| M6 | **Lakehouse lineage closure.** Every gold artifact's recorded lineage resolves to an existing silver artifact whose content hash matches; a regenerated lake from the same config is content-identical (`tests/unit/data/test_reproducibility.py` covers the second half) | `data/lakehouse/{lineage,quality,store,receipts}` |
| M7 | **Ingest determinism.** `ingest(config)` on a SYNTHETIC source produces a byte-identical bronze/silver/gold set across two runs and across two worker counts | `data/ingest`, `data/concurrent_io` |
| M8 | **Universe membership monotonicity.** Widening the filter cannot remove a name; a delisted name never appears after its delisting date | `data/universe`, `data/security_master` |
| M9 | **No-network structural guard.** Extend `tests/unit/data/test_no_network.py` from two `open(...).read()` substring checks into an AST walk over `quant_fund.data.adapters` and `quant_fund.data.sources` asserting no import of `requests`/`httpx`/`urllib.request`/`socket` outside the explicitly networked vendor adapters (which are `network`-marked) | `data/adapters`, `data/sources` |

**Parser fuzzing (Atheris).** New `tests/fuzz/` (excluded from `testpaths`,
run by a dedicated workflow — the same lane-separation pattern already used for
`tests/fx1`):

- `fuzz_normalize_ohlcv.py` — `FuzzedDataProvider` builds rows with hostile
  field types; oracle = "raises `SourceError` or returns a frame satisfying the
  OHLC envelope, magnitude bound, and timestamp ordering". Any other outcome is
  a defect.
- `fuzz_pit_gates.py` — oracle = "raises `PointInTimeError` or returns a frame
  with valid PIT columns and `available_time ≥ event_time`".
- `fuzz_canonical_json.py` — oracle = "total, deterministic, and
  order-independent over dict key insertion order; non-finite floats handled per
  the documented `_json_sort_key` contract".
- `fuzz_parquet_adapter.py` — truncate/corrupt bytes of a committed fixture
  parquet; oracle = "raises, never returns a partial frame silently".
- Seed corpus `tests/fuzz_corpus/` of real-shaped weird inputs (§2.6).
- CI: short PR-scoped run (30–60 s per target) + nightly batch that grows the
  corpus, following the ClusterFuzzLite mode split. Store corpora as CI
  artifacts.
- Bridge option: `atheris.instrument_all()` plus the existing Hypothesis
  strategies from §6.1 as the input decoder gets coverage-guided search over
  *structured* inputs for free, reusing the same oracles. Do this for
  `normalize_ohlcv` first.

### 6.5 Phase 5 — stability, golden masters, and reproducibility

1. **Tolerance registry.** Add `tests/support/tolerances.py` with named
   constants and a one-line rationale + derivation each, e.g.
   `DOT_RTK = 1e-12  # κ≈1 × n≤1e3 × u=1.1e-16, ×64 safety`,
   `CRPS_QUANTILE_RIEMANN_RTK`, `HAC_LRW_RTK`, `COV_BACKWARD_RTK`. Replace
   ad-hoc literals in the property lane only (260 files use `pytest.approx`;
   do not sweep them all). Document the registry in `docs/MATH_SPEC.md`.
2. **Stability battery** (§2.3): cancellation, order-independence, conditioning
   sweep, overflow/underflow frontier. New `tests/property/test_numeric_stability.py`.
3. **ULP assertions where bit-identity is the contract.** `native.reference`
   already distinguishes `_bit_exact` from `_close`; express the `_close` side
   with `np.testing.assert_array_max_ulp(..., maxulp=N)` for the kernels where
   an N-ULP bound is derivable, so the claim is "≤ N ULP" rather than "within
   some rtol".
4. **Counterexample pinning discipline (process, zero code).** When a nightly
   or local run finds a failure: commit the shrunk example as
   `@example(...)` on the property, and paste the `# Reproduce failure:` blob
   from `print_blob` into the test docstring. Then extend `print_blob=True`
   beyond the single file that has it — it is one kwarg in `adversarial_settings()`
   and it is the difference between a reproducible CI failure and a shrug.
5. **Nightly coverage of the whole property lane (G5).** Change
   `property-nightly.yml` to run `tests/property tests/formal` at the nightly
   profile instead of six named files, with a per-file budget guard so runtime
   stays bounded. The 24 currently-excluded files include the LOB, calendar,
   PIT, order-book, reality and robustness properties — the ones most likely to
   have a boundary bug that 20 examples misses.
6. **Golden masters out of CI YAML (G12).** Move the receipt/notebook/scorecard
   shape assertions in `ci.yml`'s `smoke` job into a pytest module
   (`tests/regression/test_research_receipt_shape.py`) that CI then calls, so
   the assertions are locally runnable, coverage-measured, and reviewable.
   Keep the CI step as a one-line invocation.
7. **Optional snapshot plugin.** If frame-level snapshots become desirable
   (lakehouse quality reports, `doctor` output), add `pytest-regtest` — it has
   a polars handler and explicit `rtol`/`atol`, which syrupy lacks. Do not add
   syrupy for numeric payloads.
8. **Determinism lint.** A structural test asserting no property test reads
   wall-clock time or unseeded randomness (`time.time`, `datetime.now`,
   `np.random.rand` without a `default_rng`) — the `Flaky` failure mode that
   Hypothesis warns about but does not prevent.

### 6.6 Sequencing and cost

| Phase | Contents | Effort | Blocking on |
|---|---|---|---|
| 0 | Restore `pyproject.toml`/`uv.lock`; drop stray mirrors; explicit `.hypothesis` ignore | S | — |
| 1 | `_numeric_strategies.py`, `tests/property/conftest.py`, INVARIANTS docstring convention | S | Phase 0 |
| 2a | `metrics/scoring` (S1–S22), `metrics/probability` (P1–P11) | M | Phase 1 |
| 2b | `metrics/risk`, `metrics/conformal` (R1–R9) | M | Phase 1 |
| 2c | `metrics/evalues`, `anytime_fdr`, `e_detectors` (E1–E9) | M | Phase 1 |
| 2d | `energy_score`, `density_forecast`, `twosample` (N1–N8) | S | Phase 1 |
| 2e | `serial`, `fractal`, `entropy`, `hac`, `bootstrap`, `purged_cv` (T1–T10) | M | Phase 1 |
| 2f | `snooping`, `overfitting` extensions (D1–D7), `utils/numeric` (U1–U5) | S | Phase 1 |
| 3 | Mutation baseline test + weekly CI job over tier-1 set | M | Phase 2a (so `scoring`/`probability` have mutants worth killing) |
| 4 | Data-engine MRs (M1–M9) then Atheris harnesses + corpora | M | Phase 1 |
| 5 | Tolerance registry, stability battery, ULP assertions, nightly widening, receipt-shape golden masters, counterexample discipline | M | Phase 1 |

Ordering rationale: Phase 2a before Phase 3 because a mutation run on
`metrics/scoring` is only informative once properties exist to kill the
mutants — otherwise the score measures the known-answer tests, which is a
different (and less useful) number.

### 6.7 Non-goals

- **No Sharpe/Sortino/Calmar/P&L/NAV headlines**, in tests or in this
  document's future revisions. Where a property must reason about a
  Sharpe-like quantity (PSR/DSR in §6.2.7), it is tested as a *statistic of a
  statistic* under the honesty-contract family, never surfaced as a result.
  `tests/unit/research/test_bench_forbidden_metrics.py` and
  `tests/fx1/test_honesty_inheritance.py` are the drift guards; the R9
  structural test in §6.2.3 extends the same idea to the estimator layer.
- **No live-trading or broker-connectivity tests.** Nothing here touches a
  venue. `SimulatedBroker` is the only broker, and the TLA+/Z3 lane stays the
  authority on order semantics.
- **No 100% mutation-score target.** See §2.5/§6.3.
- **No global line-coverage increase as a goal.** The 80% floor and the five
  proofcore floors are held, not raised, by this plan; a `metrics` floor may be
  added *after* Phase 2 lands, at the then-measured level, ratchet-only.
- **No market data in any new test.** SYNTHETIC inputs only, labelled, per
  honesty-contract rule 2.

---

## 7. Cross-references

- `AGENTS.md` — gates, honesty contract, lane conventions (`tests/fx1` outside
  `testpaths`; `make fx1-test`).
- `docs/FORMAL_VERIFICATION.md` — TLC/Z3/stateful lane this document builds on
  rather than duplicates.
- `docs/VALIDATION.md` — walk-forward/purge/embargo/CPCV contracts; T10 in
  §6.2.6 is the generated-input version of the per-group-purge rule.
- `docs/MATH_SPEC.md` — where the tolerance registry (§6.5) and the per-module
  INVARIANTS catalogs should be summarized.
- `docs/SOTA_CANON_ROADMAP_2026_09.md` §2.4 — items (4) chaos/fault injection,
  (5) Hypothesis CI-profile pinning and (6) mutation testing on
  `verify.py`/`catalog.py`. Items (5) and (4) are **done** (root conftest
  registers the `ci` profile; `test_ingest_fault_injection.py` exists); item (6)
  is extended by §6.3.
- `docs/SOTA_GAP_ANALYSIS.md`, `docs/REPO_IMPROVEMENT_PLAN.md`,
  `docs/ULTRAPLAN_FRONTIER.md` — the frontier line "mutation testing of money
  paths" is what §6.3 operationalizes.
- `docs/adr/0008-forbidden-metric-key-scan.md`,
  `docs/adr/0004-fail-closed-defaults.md`,
  `docs/adr/0003-fx1-test-lane-separation.md` — invariants the property work
  must preserve.
- `scripts/mutation_score.py`, `tests/property/mutation_scores.json`,
  `tests/property/_profiles.py`, `tests/support/session_cache.py`,
  `tests/unit/test_quality_ratchet.py` — the existing machinery this plan
  extends.

---

## 8. Reproducing the audit numbers

```bash
# Lane topology (tracked files only)
git ls-files "tests/unit/*.py" -r | wc -l        # 851
git ls-files "tests/property/test_*.py" | wc -l  # 30
git ls-files "tests/fx1/test_*.py" | wc -l       # 27

# Test-function counts
git grep -c "def test_" -- tests | awk -F: '{s+=$2} END {print s}'   # ~7502
git grep -c "def test_" -- tests/fx1 | awk -F: '{s+=$2} END {print s}' # 248

# Hypothesis footprint
git grep -l "from hypothesis import\|import hypothesis" -- tests | wc -l  # 42
git grep -l "@given" -- tests | wc -l                                     # 39
git grep -c "@given" -- tests | awk -F: '{s+=$2} END {print s}'           # ~358
git grep -c "@given" -- tests/property/test_order_book_metrics_identity.py # 226
git grep -l "hypothesis.extra.numpy" -- tests | wc -l                     # 0

# Metrics property coverage (the 4/58 finding)
git ls-files "src/quant_fund/metrics/*.py" | wc -l                        # 58
for f in $(git grep -l "from hypothesis import" -- tests); do
  git grep -oh "quant_fund\.metrics\.[a-z_0-9]*" -- "$f"
done | sort -u    # analytics, cross_section, overfitting, returns  => 4

# Marker usage
for m in network slow synthetic native smoke perf_full; do
  printf "%s=%s\n" "$m" "$(git grep -l "mark.$m" -- tests | wc -l)"
done   # 1 6 22 3 6 3

# Tolerance practice
git grep -l "pytest.approx" -- tests | wc -l          # 260
git grep -l "assert_allclose" -- tests | wc -l        # 37
git grep -l "assert_array_max_ulp" -- tests | wc -l   # 0

# Settings knobs
for k in "deadline=None" "derandomize=True" "print_blob" \
         "suppress_health_check" "@example"; do
  printf "%s=%s\n" "$k" "$(git grep -l -- "$k" -- tests | wc -l)"
done   # 27 6 1 7 0

# Working-tree divergence (G1)
git diff HEAD -- pyproject.toml | head -60
uv run pytest tests/property/test_native_kernels.py --collect-only -q
#   ERROR ... 'native' not found in `markers` configuration option
```

---

## 9. References

**Property-based testing**
- K. Claessen, J. Hughes. *QuickCheck: A Lightweight Tool for Random Testing of
  Haskell Programs.* ICFP 2000, 268–279. https://www.cs.tufts.edu/comp/150FP/archive/john-hughes/quick.pdf
- D. R. MacIver, Z. Eide, C. Grey. *Hypothesis: A New Approach to Software
  Testing.* Journal of Open Source Software 4(37):1891, 2019.
- Hypothesis documentation — numpy extra. https://hypothesis.readthedocs.io/en/latest/numpy.html
- H. Goldstein. *Property-based testing and fuzzing in practice* (dissertation).
  https://harrisongoldste.in/papers/dissertation.pdf
- N. Teodorescu. *Fuzzing vs property testing.* 2018. https://www.tedinski.com/2018-12-11/fuzzing-and-property-testing.html

**Metamorphic testing**
- T. Y. Chen, F.-C. Kuo, H. Liu, P.-L. Poon, D. Towey, T. H. Tse, Z. Q. Zhou.
  *Metamorphic Testing: A Review of Challenges and Opportunities.* ACM Computing
  Surveys 51(1):4, 2018. doi:10.1145/3143561
- S. Segura, G. Fraser, A. B. Sánchez, A. Ruiz-Cortés. *A Survey on Metamorphic
  Testing.* IEEE TSE 42(9):805–824, 2016.
- T. Y. Chen et al. *Metamorphic testing 20 years later.* ICSE 2018 Companion.
  doi:10.1145/3183440.3183468
- *Metamorphic Relation Generation: State of the Art and Visions for Future
  Research.* arXiv:2406.05397.
- U. Kanewala, J. M. Bieman, A. Ben-Hur. *Predicting metamorphic relations for
  testing scientific software: a machine learning approach using graph kernels.*
  STVR 26(3):245–269, 2016.

**Scoring rules and evaluation methodology**
- T. Gneiting, A. E. Raftery. *Strictly Proper Scoring Rules, Prediction, and
  Estimation.* JASA 102(477):359–378, 2007. doi:10.1198/016214506000001437
- W. Ehm, T. Gneiting, A. Jordan, F. Krüger. *Of quantiles and expectiles:
  Consistent scoring functions, Choquet representations and forecast rankings.*
  JRSS-B 78(3):505–562, 2016.
- T. Fissler, J. F. Ziegel. *Higher order elicitability and Osband's principle.*
  Ann. Math. Statist. / AoS-adjacent; joint (VaR, ES) elicitability, 2016.
- J. Bracher, E. L. Ray, T. Gneiting, N. G. Reich. *Evaluating epidemic
  forecasts in an interval format.* PLOS Comput. Biol. 17(2):e1008618, 2021.
- T. Gneiting, R. Ranjan. *Combining predictive distributions.* Electron. J.
  Stat. 7:1747–1782, 2013. doi:10.1214/13-EJS823
- G. J. Székely. *Statistics on the energy distance.* InterStat, 2003.
- G. J. Székely, M. L. Rizzo. J. Statist. Plann. Inference 143(8):1249–1272,
  2013. arXiv:1210.3927
- J. E. Matheson, R. L. Winkler. *Scoring rules for continuous probability
  distributions.* Management Science 22(10):1087–1096, 1976.
- *Why scoring functions cannot assess tail properties.* arXiv:1905.04233.

**Floating point and numerical stability**
- N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM,
  2002. https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/
- N. J. Higham. *What is Numerical Stability?* 2020.
  https://nhigham.com/2020-08-04/what-is-numerical-stability/
- N. J. Higham. *Testing Linear Algebra Software.*
  https://nhigham.com/wp-content/uploads/2023/10/high97t.pdf
- LAPACK Working Note 41 — *LAPACK Installation and Testing Guide.*
  https://netlib.org/lapack/lawnspdf/lawn41.pdf
- NumPy testing support (`assert_allclose`, `assert_array_max_ulp`,
  `assert_array_almost_equal_nulp`). https://numpy.org/doc/stable/reference/routines.testing.html
- W. Kahan. *Kahan summation algorithm* (error analysis, condition-number
  dependence). https://en.wikipedia.org/wiki/Kahan_summation_algorithm
- J. D. Cook. *Summing an array of floating point numbers.* 2019.
  https://www.johndcook.com/blog/2019-11-05/kahan/
- G. Singh et al. *Floating-point error estimation via automatic
  differentiation (Clad).* SIAM UQ 2022.
  https://compiler-research.org/assets/presentations/G_Singh-SIAMUQ22_FP_Error_Estimation.pdf

**Golden-master / snapshot testing**
- syrupy. https://syrupy-project.github.io/syrupy/
- pytest-regtest — snapshots for NumPy/pandas/polars with `rtol`/`atol`.
  https://pytest-regtest.readthedocs.io/en/stable/

**Mutation testing**
- Y. Jia, M. Harman. *An Analysis and Survey of the Development of Mutation
  Testing.* IEEE TSE 37(5):649–678, 2011.
- A. J. Offutt, A. Lee, G. Rothermel, R. H. Untch, C. Zapf. *An experimental
  determination of sufficient mutant operators.* ACM TOSEM 5(2):99–118, 1996.
- G. Petrović, M. Ivanković. *State of Mutation Testing at Google.* ICSE-SEIP
  2018, 163–171. https://research.google/pubs/state-of-mutation-testing-at-google/
- G. Petrović, M. Ivanković, B. Kurtz, P. Ammann, R. Just. *Practical Mutation
  Testing at Scale: A View from Google.* IEEE TSE, 2021. doi:10.1109/TSE.2021.3107634
- R. Just, A. Jalali, M. D. Ernst. *Are Mutants a Valid Substitute for Real
  Faults in Software Testing?* FSE 2014. doi:10.1145/2635868.2635929
- M. Papadakis, S. Shin, J. Yoo, D. Bae. *Are Mutation Scores Correlated with
  Real Fault Detection?* ICSE 2018. doi:10.1145/3180155.3183519
- *What it would take to use mutation testing in industry.* ICSE-SEIP 2021.
  doi:10.1109/ICSE-SEIP52600.2021.00036
- mutmut. https://github.com/boxed/mutmut
- Cosmic Ray. https://github.com/sixty-north/cosmic-ray
- mutation-gate (diff-scoped, complexity-triggered pre-commit gate).
  https://github.com/mikemartincode/mutation-gate
- CircleCI. *What is mutation testing?* https://circleci.com/blog/what-is-mutation-testing/

**Fuzzing**
- B. P. Miller, L. Fredriksen, B. So. *An Empirical Study of the Reliability of
  UNIX Utilities.* CACM 33(12):32–44, 1990.
- V. J. M. Manès, H. Han, C. Han, S. K. Cha, M. Egele, E. J. Schwartz, M. Woo.
  *The Art, Science, and Engineering of Fuzzing: A Survey.* IEEE TSE
  47(11):2312–2331, 2021.
- Atheris — coverage-guided Python fuzzer. https://github.com/google/atheris
- ClusterFuzzLite — continuous fuzzing in CI (PR / batch / coverage / prune
  modes). https://google.github.io/clusterfuzzlite/

**Coverage adequacy**
- T. Ahmed, R. Gopinath, et al. *Can Testedness be Effectively Measured?* FSE
  2016. https://rahul.gopinath.org/resources/fse2016/ahmed2016can.pdf
- L. Inozemtseva, R. Holmes. *Coverage is not strongly correlated with test
  suite effectiveness.* ICSE 2014.
- A. S. Namin, J. H. Andrews. *The influence of size and coverage on test suite
  effectiveness.* ISSTA 2009, 57–68.
- *Mind the Gap: The Difference Between Coverage and Mutation Score Can Guide
  Testing Efforts.* arXiv:2309.02395.
