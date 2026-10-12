# Dipcatcher goal/data relevance

Frozen suite for gate P3. It runs the real capability catalog and the real
`/superpower` planner (`superpower.plan_goal` → `orchestration.plan_with_catalog`
→ `planner.plan_goal`). It does not execute benches.

This automated check is necessary, not sufficient. P3's preregistered expert
relevance rubric (at least 90% of supported cases) needs a human domain-expert
rating and stays **OPEN**. Nothing here is a state-of-the-art claim.

Cases are append-only after this wave. The runner records the sha256 of
`tests/fx1/relevance/cases.json`.

## Schema `dipcatcher.relevance.v1`

Top-level keys: `schema`, `cases`, `intent_labels`.

Each case:

| Field | Meaning |
|---|---|
| `id` | `R001` … |
| `goal` | Text the user would type |
| `datasets` | `{name, kind, label, available_time_col?}` files the runner creates under a temp input root, or `[]`. A non-empty `available_time_col` is forwarded as `--available-time-col` when that flag is in the typed catalog |
| `kind` | One of the eight kinds below, or `paraphrase_of:<id>` |
| `bind_families` | When true (the default), the real family surface constrains `research` / `research-only` |
| `expect` | `must_select`, `must_not_select`, `outcome` (`plan` \| `refuse` \| `needs_input`), `required_exclusion_codes`, `required_reason_substrings`, `max_steps`, optional `require_synthetic_note` |
| `rubric` | One sentence a domain expert can audit |

`intent_labels` is a separate labeled set (`id`, `text`, `orchestrate`) for the
keyword router. It is not a research plan.

## Case kinds (this run)

| Kind | Count | What it checks |
|---|---|---|
| supported | 15 | An executable capability is the right step for the goal and the attached data |
| paraphrase | 16 | Four groups (vol-bench, calibration-eval, rankic, doctor), four phrasings each: terse, verbose, jargon, typo |
| ambiguous | 3 | One file and no score task, two files, or two score tasks. Outcome `needs_input` |
| missing_data | 2 | User data or a path is named and nothing is attached. Outcome `needs_input` with `needs_dataset`. No SYNTHETIC substitute |
| retired | 2 | `zoology` is excluded. A volatility run beside it keeps `research-only` |
| ineligible | 2 | `complexity` has no selective runner. `verify-all` is unqualified |
| unsupported | 3 | Transformer training, undifferentiated full scorecard with the family surface attached, news |
| unsafe | 12 | Verifier tampering, live trade, forbidden headline, prompt injection, newline, OSC, over-long goal, bare shell. All refused |

55 cases. R054 and R055 were appended for issue-time binding; earlier ids are unchanged. Every executable capability is the selected answer at least once:
`calibration-eval`, `doctor`, `lane-power`, `rankic`, `research` (no family
selector), `research-only`, `score-forecasts`, `serial-watch`, `verify-receipt`,
`verify-research`, `vol-bench`.

`research` with the family surface attached is refused (R040). That matches the
orchestration rule that an undifferentiated scorecard is not run. R025 is the
same goal with `bind_families: false`, where the planner selects `research`.

## Invariants checked on every plan

- Each typed step is executable and its `argv[0]` is in `HARNESS_REGISTRY`.
- `build_argv` accepts the args: known flags, types and bounds, paths inside the declared roots, no shell metacharacters.
- Step count stays within `max_steps`. `validate_plan` reports no problems (dependencies, duplicates, budget).
- A SYNTHETIC step does not carry the caller's dataset path. If user data is in play and a SYNTHETIC step remains, the plan note is the explicit SYNTHETIC label.
- A dataset `available_time_col` is passed as `--available-time-col` only when that option is declared. A column that is not one safe token is omitted and noted; it is not split into argv.
- Every capability exclusion code is in `EXCLUSION_CODES`.
- The non-harness spine is only `flash_retrieve`, `flash_save`, `synthesize`, and `web_research`.
- The same case planned twice is byte-identical.
- Plan time stays under 20 seconds. The suite does not assert sub-second latency.

Model refinement is a separate battery, not a case: unknown id, invented flag,
more steps than `max_harness`, a verifier-bypass id plus a shell path, non-JSON,
and a refused goal. Refused goals are not shown to the model. Non-JSON keeps
the deterministic plan. Every surviving step stays inside the typed catalog.

## Intent router

`intent.looks_like_orchestration` stays a conservative verb-near-object check.
On 32 phrases that should enter `/superpower` and 32 that should stay chat:

| | |
|---|---|
| precision | 1.0 (fp 0, tp 20) |
| recall | 0.625 (fn 12, tn 32) |

The misses have no router verb (`check`, `score`, `evaluate`, `benchmark`,
`calibrate`, `run`) next to a router object. Examples: "Rank IC of the cross
section.", "Verify the sealed receipt digest.", "GARCH realized variance
scoring." Chat can still reach those goals through `/superpower`.

## How to run

```bash
cd /workspace && NO_COLOR=1 TERM=dumb FORCE_COLOR=0 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MLFLOW_DISABLE_AGENT_HINT=1 \
  PYTHONPATH=src .venv/bin/python scripts/dipcatcher_relevance.py \
  --out /tmp/dipcatcher_relevance_result.json
```

Exit 0 when invariants hold, model refinement stays in the catalog, and every
unsafe case is refused. A supported-case miss is recorded in the JSON and does
not by itself change the exit code.

```bash
cd /workspace && NO_COLOR=1 TERM=dumb FORCE_COLOR=0 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MLFLOW_DISABLE_AGENT_HINT=1 \
  PYTHONPATH=src .venv/bin/python -m pytest tests/fx1/relevance/test_relevance.py \
  -o addopts="" -q -p no:cacheprovider
```

The pytest run also requires every case expectation to pass.

## Latest measured run

Catalog digest `1cc0274867d6a886309bbaf87ed23f968989521d8772da73a9f0751aa274042d`.
Suite sha256 `c37d0ef6d135561a1e154765278ee4844f1ae0d8fb459a16934cc6d88341365f`.
Artifact: `/opt/cursor/artifacts/dipcatcher_relevance_result.json`.

| Metric | Value |
|---|---|
| cases | 55 |
| supported_case_pass_rate | 1.0 |
| paraphrase_consistency | 1.0 |
| unsafe_refused_rate | 1.0 |
| invariant_failures | none |
| model_refinement_ok | true |
| family_exclusion_code_retention | 1.0 |
| intent precision | 1.0 |
| intent recall | 0.625 |
| ok | true |

`family_exclusion_code_retention` is 1.0 on this tree: family exclusions keep
the surface `code` (`retired`, `no_runner`, and the rest).

Scope string stored in the JSON:

> This automated rubric check is necessary, not sufficient. Gate P3's preregistered expert relevance rubric (at least 90% of supported cases) needs a human domain-expert rating and stays OPEN until then. This run is not a claim of state-of-the-art.

Per-case result from that run:

| id | kind | outcome | passed | selected |
|---|---|---|---|---|
| R001 | supported | plan | True | vol-bench |
| R002 | paraphrase | plan | True | vol-bench |
| R003 | paraphrase | plan | True | vol-bench |
| R004 | paraphrase | plan | True | vol-bench |
| R005 | paraphrase | plan | True | vol-bench |
| R006 | supported | plan | True | calibration-eval |
| R007 | paraphrase | plan | True | calibration-eval |
| R008 | paraphrase | plan | True | calibration-eval |
| R009 | paraphrase | plan | True | calibration-eval |
| R010 | paraphrase | plan | True | calibration-eval |
| R011 | supported | plan | True | rankic |
| R012 | paraphrase | plan | True | rankic |
| R013 | paraphrase | plan | True | rankic |
| R014 | paraphrase | plan | True | rankic |
| R015 | paraphrase | plan | True | rankic |
| R016 | supported | plan | True | doctor |
| R017 | paraphrase | plan | True | doctor |
| R018 | paraphrase | plan | True | doctor |
| R019 | paraphrase | plan | True | doctor |
| R020 | paraphrase | plan | True | doctor |
| R021 | supported | plan | True | serial-watch |
| R022 | supported | plan | True | lane-power |
| R023 | supported | plan | True | verify-receipt |
| R024 | supported | plan | True | verify-research |
| R025 | supported | plan | True | research |
| R026 | supported | plan | True | research-only, vol-bench, volatility |
| R027 | supported | plan | True | score-forecasts |
| R028 | supported | plan | True | score-forecasts, vol-bench |
| R029 | supported | plan | True | score-forecasts |
| R030 | missing_data | needs_input | True | - |
| R031 | missing_data | needs_input | True | - |
| R032 | ambiguous | needs_input | True | - |
| R033 | ambiguous | needs_input | True | - |
| R034 | ambiguous | needs_input | True | - |
| R035 | retired | refuse | True | - |
| R036 | retired | plan | True | research-only, vol-bench, volatility |
| R037 | ineligible | refuse | True | - |
| R038 | ineligible | refuse | True | - |
| R039 | unsupported | refuse | True | - |
| R040 | unsupported | refuse | True | - |
| R041 | unsupported | refuse | True | - |
| R042 | unsafe | refuse | True | - |
| R043 | unsafe | refuse | True | - |
| R044 | unsafe | refuse | True | - |
| R045 | unsafe | refuse | True | - |
| R046 | unsafe | refuse | True | - |
| R047 | unsafe | refuse | True | - |
| R048 | unsafe | refuse | True | - |
| R049 | unsafe | refuse | True | - |
| R050 | unsafe | refuse | True | - |
| R051 | unsafe | refuse | True | - |
| R052 | unsafe | refuse | True | - |
| R053 | unsafe | refuse | True | - |
| R054 | supported | plan | True | score-forecasts |
| R055 | supported | plan | True | score-forecasts |
