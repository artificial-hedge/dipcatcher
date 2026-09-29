# 17 — API Design, Versioning & Documentation Excellence

> Lane: **API DESIGN, VERSIONING & DOCUMENTATION EXCELLENCE.** Status doc —
> research + audit + adoption plan. No source, config, or workflow file is
> modified here; every snippet below is a proposal to land in a separate PR.
>
> Scope of audit: `src/fx1/**` (65 modules), `src/quant_fund/__init__.py` +
> `quant_fund.public` facade, `pyproject.toml` (name/version/build/ruff/mypy/
> pytest), `CHANGELOG.md`, `mkdocs.yml` + `docs/**` (155 files) + `docs/api/**`,
> `examples/**` + `tests/examples/test_examples_run.py`,
> `tests/unit/test_public_api.py` + `public_api_snapshot.txt`,
> `tests/fx1/test_docs_drift.py`, `docs/FX1_API_STABILITY.md`.
>
> Governing contract: `docs/FX1_API_STABILITY.md` — `fx1.__version__` is the
> canonical semver; `pyproject.toml` reads it via `[tool.hatch.version]
> path = "src/fx1/__init__.py"`. This doc strengthens that contract; it
> proposes no weakening of the honesty rules in `AGENTS.md`.
>
> **Date:** 2026-09-28. Tree state caveat: the working tree was observed
> mid-restore during the sibling audit (`docs/SOTA/19-code-quality.md` §0) —
> `pyproject.toml` flapped between a stale `dipcatcher 1.0.0` variant and the
> HEAD `fx-1` + dynamic-version variant. All pyproject findings below were
> re-verified against the HEAD (369-line, `name = "fx-1"`, `dynamic =
> ["version"]`) content. A stale `dipcatcher-1.0.0.dist-info` *was* observed
> installed in `.venv` alongside `fx_1-0.4.0.dist-info` during the first
> audit pass; a **re-check at 21:58 IST found it gone** (a re-sync cleaned
> it) — `importlib.metadata.version("fx-1") == fx1.__version__ == "0.4.0"`
> now holds and `dipcatcher` is no longer resolvable as a distribution. G10
> is therefore recorded as *transient-but-unprevented*, not as a live defect.
> The `fx1.data.sources` `RecursionError` (G1) was **re-confirmed live** in
> the same re-check.

---

## 1. Verdict up front

The repo is **at or near SOTA on version governance and changelog discipline,
materially behind SOTA on API-surface testing for `fx1`, docstring/doctest
coverage, and generated API docs for the product package**, and carries
**one live public-API bug** (lazy `fx1.data.sources` attribute access raises
`RecursionError`).

**Strong (at or above typical OSS SOTA):**

- **Single-source versioning.** `fx1.__version__ = "0.4.0"` is the only
  literal; `pyproject.toml` declares `dynamic = ["version"]` with
  `[tool.hatch.version] path = "src/fx1/__init__.py"`, and
  `quant_fund/__init__.py` re-exports it (`from fx1 import __version__ as
  __version__`) with a comment stating the package "does not keep a second
  literal". `tests/fx1/test_cli.py` asserts `fx1 --version` echoes it. This
  is exactly the "one canonical version" pattern; drift is structurally
  hard.
- **Dual-versioning awareness for model vs package.**
  `docs/FX1_API_STABILITY.md` already separates package semver from
  checkpoint tags (`fx-1.vX.Y` + model card) and states that receipts embed
  hashes, not versions — the recommended MLOps split (monotonic/content-
  addressed artifact identity + semantic family tags; §2.2).
- **A real typed facade with a snapshot test.** `quant_fund.__init__` is a
  PEP 562 lazy facade (`__getattr__` + `globals()` caching + `__dir__`),
  `quant_fund.public` bans `Any`, and `tests/unit/test_public_api.py`
  renders a signature-level snapshot (`public_api_snapshot.txt`) and diffs
  it — including a subprocess test that `import quant_fund` does *not* load
  the research stack, and a `py.typed == "partial\n"` assertion. This is
  stronger than most public OSS libraries do.
- **Keep a Changelog compliance.** `CHANGELOG.md` declares Keep a Changelog
  1.1.0 + SemVer 2.0.0, names `fx1.__version__` as the version source, and
  uses `## [Unreleased]` with `### Fixed / Added / Changed` sections.
- **Docs-as-executable-tests lane.** `examples/01..05_*.py` run in isolated
  subprocesses with **outbound sockets blocked** and a 120 s timeout
  (`tests/examples/test_examples_run.py`), gated by `make examples` (ruff +
  mypy + runner). Offline-first executable examples are the strongest form
  of doc testing and this lane already exists.
- **Docs/CLI drift guard.** `tests/fx1/test_docs_drift.py` checks that every
  `fx1 …` command, `make …` target, and `from fx1.* import …` path in
  `README.md` + `docs/FX1*.md` actually exists.
- **Strict docs build.** `mkdocs build --strict` in CI (`.github/workflows/
  docs.yml`) with mkdocstrings (Google style), `filters: ["!^_"]`, and
  `validation.* : warn` — warnings become failures under `--strict`.

**Gaps (ranked by leverage):**

| # | Gap | Impact | Fix |
|---|---|---|---|
| G1 | `fx1.data.__getattr__("sources")` recurses infinitely: `from fx1.data import sources`, `fx1.data.sources` after bare `import fx1.data`, and `from fx1.data import *` **all raise `RecursionError`** (reproduced live). Any `__all__`-driven tool (mkdocstrings, griffe, `dir()`, star-import) trips it. `"sources"` is listed in `fx1.data.__all__`, so this is a broken *promised* export. | Public-API bug on a stable-documented surface | §5.1 |
| G2 | No API-surface snapshot or breaking-change gate for `fx1` — the package `docs/FX1_API_STABILITY.md` promises stability for. `quant_fund` has `public_api_snapshot.txt`; `fx1` has nothing equivalent, and no `griffe check` anywhere. | The stability policy is prose-only; regressions are invisible until users hit them | §5.2 |
| G3 | Deprecation policy is unenforceable as written: **zero** `DeprecationWarning` / `@deprecated` / `warnings.warn` sites in `src/fx1` *and* `src/quant_fund`; pytest globally `ignore::DeprecationWarning`, so even a correct deprecation would be silenced in tests. No helper exists to standardize one. | "At least one minor version" promise has no machinery | §5.3 |
| G4 | Docstring coverage: `fx1` 91/358 public defs missing docstrings (**25.4%**); `quant_fund` 1294/4676 (**27.7%**). Six symbols inside `fx1` `__all__` lists lack docstrings (`compare_runs`, `issue_receipt`, `build_manifest`, `load_harness_config`, `get_spec`, `list_sources`) despite `show_if_no_docstring: true` (they render as bare signatures). No docstring gate (no ruff `D` rules, no interrogate/pydocstyle). | Generated docs and LLM-consumable docs are thin exactly where the stable surface is | §5.4 |
| G5 | Doctest coverage is **zero**: no `>>>` anywhere in `src/fx1` (65 modules) or `src/quant_fund` (670 modules); `addopts` has no `--doctest-modules`. | Docstring examples can rot silently; the examples lane covers scripts, not API docs | §5.5 |
| G6 | `fx1` ships **without a `py.typed` marker** while riding in the same wheel as `quant_fund` (which has `py.typed` = `partial`). PEP 561: type checkers ignore annotations in non-marked packages → `import fx1` is untyped for every downstream consumer, including our own mypy `packages = ["fx1", "quant_fund"]` only works because it runs from source. | Typed-API promise not visible outside the checkout | §5.6 |
| G7 | Generated API reference covers only `quant_fund`: `docs/api/*.md` contain 19 `:::` targets, **all** `quant_fund.*`; `fx1` has zero mkdocstrings pages despite being the product and the semver-governed surface. | The package whose version we guarantee has no rendered API docs | §5.7 |
| G8 | 9 of 17 audited `fx1` modules have **no `__all__`** — including `fx1.harness`, `fx1.honesty`, `fx1.mrm`, `fx1.modelcard`, `fx1.reward`, `fx1.sbom`, `fx1.doctor`, `fx1.hypotheses`, `fx1` root — yet `FX1_API_STABILITY.md` names `fx1.harness.Harness.run/list_commands`, `fx1.honesty.validate_fx1_output`, `fx1.mrm.compile_dossier` as *stable*. Public-vs-private boundary is implicit in exactly the modules with explicit guarantees. | "Undocumented = unstable" is ambiguous when the stable symbols live in modules without `__all__` | §5.8 |
| G9 | Nav drift: the `docs/SOTA/*.md` lane (15 files before this doc, 16 with it) is in neither `mkdocs.yml` nav nor `exclude_docs` → each emits an `omitted_files: warn` under `--strict` (verified: "Aborted with 47 warnings in strict mode"). The whole lane is invisible on the published site. | Strict-build warning noise; risk of a future `not_found` escalation breaking CI | §5.9 |
| G10 | Installed-env drift (transient): `.venv` was observed holding both `dipcatcher-1.0.0.dist-info` (stale name/version, no `fx1` script) and `fx_1-0.4.0.dist-info` (HEAD); a re-sync cleared it. Nothing *prevents* recurrence, and no test would notice. | Wrong `quant`/`fx1` console scripts; `importlib.metadata.version()` disagrees with `fx1.__version__` | §5.10 |
| G11 | Typed-API ergonomics underused in `fx1`: 2 `Protocol` sites (`serve.backends.InferenceBackend`, `forecast.protocol`), **0** `TypedDict`, **0** `@overload`, **0** `@runtime_checkable` across 65 modules. | Boundary contracts (JSON blobs, backend dispatch) are `dict[str, Any]`-shaped | §5.11 |

---

## 2. Practice summaries (research) + citations

### 2.1 Public/private boundaries and `__all__` discipline

- **Explicit exports beat implicit ones.** A module without `__all__` has an
  ambiguous boundary: everything not underscore-prefixed is importable, but
  nothing is *promised*. The standard discipline: every package
  `__init__.py` that is part of the public surface declares `__all__`;
  underscore-prefix anything not in it; star-imports then mean exactly
  `__all__`. `__all__` also drives tooling — mkdocstrings member selection,
  griffe's breakage analysis (non-exported names are not API), and IDE
  completion.
- **Lazy facades must cache.** PEP 562 module `__getattr__` is the standard
  for keeping `import pkg` light, but the handler must (a) avoid `from
  pkg import submodule` (which re-enters `__getattr__` via
  `importlib._handle_fromlist`'s `hasattr` probe — a classic infinite-
  recursion trap) and (b) store the resolved value in `globals()` so the
  second access is free and `hasattr` stabilizes. `quant_fund/__init__.py`
  does (b) correctly; `fx1/data/__init__.py` does neither (G1).
  - CPython `importlib._bootstrap._handle_fromlist`: for each name in a
    `from … import …` list, it checks `hasattr(module, x)` before importing
    the submodule — so a `__getattr__` that itself does `from pkg import x`
    recurses.
- **API-diff tooling.** Griffe (the engine under mkdocstrings) ships
  `griffe check <pkg> [--against <git-ref>] [--search src]`, exiting non-zero
  on breaking changes (removed/renamed objects, changed parameter kinds or
  defaults, changed base classes). Recommended CI shape:
  `uvx griffe check --search src --format github fx1` with `fetch-depth: 0`,
  compared against the latest tag or `origin/$base_ref`.
  - <https://mkdocstrings.github.io/griffe/guide/users/checking/>
  - <https://github.com/mkdocstrings/griffe> ("Check for API breaking
    changes", `griffe check --search src mypackage`)
- **Snapshot tests complement diffs.** A checked-in signature listing
  (`public_api_snapshot.txt`) makes *every* surface change a reviewable diff
  even when it is technically non-breaking — the pattern this repo already
  uses for `quant_fund.public` (`tests/unit/test_public_api.py::
  test_public_api_snapshot`). Extending it to `fx1` is the cheapest way to
  make `FX1_API_STABILITY.md` machine-checked.
- **Hyrum's Law / stability tiers.** Publicly documented guarantees should
  name concrete symbols (as `FX1_API_STABILITY.md` does) and everything
  else should be explicitly declared unstable. The doc's "underscore-prefixed
  *or undocumented*" rule is good but only decidable if every stable module
  actually documents its boundary via `__all__` (G8).

### 2.2 Semantic versioning for packages *and* ML artifacts

- **SemVer 2.0.0** for the code package: MAJOR = breaking API, MINOR =
  additive, PATCH = fixes. <https://semver.org/spec/v2.0.0.html>
- **Dual-track model versioning** is the converged MLOps practice: a
  registry assigns **monotonic integers** (`v1, v2, …`) per registration for
  audit/rollback, while **semantic tags** (`fx-1.v2.1`) denote model-family
  milestones (new base model / architecture = major; new fine-tune data or
  adapter = minor; config/patch = patch). Artifact identity is
  **content-addressed** (hash), artifacts are **immutable** (every change =
  new version, never overwrite), and mutable pointers are **aliases**
  (`@champion`, `@production`) that decouple serving code from version
  numbers. This repo already matches the spirit: receipts hash inputs/outputs
  (`receipt_sha256`), `FX1_API_STABILITY.md` says checkpoints would be
  `fx-1.vX.Y` with model cards, and no checkpoint ships in-repo.
  - MLflow Model Registry docs (versions auto-increment; aliases like
    `models:/MyModel@champion`): <https://mlflow.org/docs/latest/ml/model-registry/>
  - MLflow, "AI Model Registry Management Checklist": monotonic build IDs
    for registry versions; semver reserved for model families; write-once
    artifact storage enforced at API level; `parent_run_id` lineage; dataset
    references by hash, not path:
    <https://mlflow.org/articles/ai-model-registry-management-checklist/>
  - MLflow, "Shared Model Registry: The Backbone of MLOps Governance":
    immutable versioned store + metadata (owner, data snapshot, commit hash,
    metrics) per version: <https://mlflow.org/articles/role-of-shared-model-registry/>
  - Atlan, "How To Version AI Models for Reproducibility and Governance"
    (2026): stage lifecycle (experimentation → validation → staging →
    production → archive), explicit approvals, LLM extension to prompts/
    adapters/retrieval configs: <https://atlan.com/know/ai-model-versioning-best-practices/>
  - Introl, "Model Versioning Infrastructure" (2025): semver mapping for
    LLM artifacts (major = different base model; minor = fine-tune on new
    data; patch = config), adapters versioned independently from base models:
    <https://introl.com/blog/model-versioning-infrastructure-mlops-artifact-management-guide-2025>
- **Distribution name vs import name vs version source must agree.** Here:
  distribution `fx-1`, imports `fx1` + `quant_fund`, version from
  `src/fx1/__init__.py`, both packages in one wheel
  (`[tool.hatch.build.targets.wheel] packages = ["src/fx1",
  "src/quant_fund"]`). A CI assertion that
  `importlib.metadata.version("fx-1") == fx1.__version__` closes the loop
  against stale installs (G10).

### 2.3 Documentation standards: style, generation, coverage gates

- **One docstring style, declared once.** Google vs numpydoc is a taste
  call; consistency is the requirement. Griffe (hence mkdocstrings) parses
  `google`, `numpy`, `sphinx`, or `auto`; the Griffe authors recommend
  Google style as "the most markup-agnostic". This repo already declares
  `docstring_style: google` in `mkdocs.yml` — keep it, and enforce it in
  lint so mixed styles can't creep in.
  - <https://mkdocstrings.github.io/griffe/docstrings/>
  - <https://mkdocstrings.github.io/griffe/guide/users/recommendations/docstrings/>
  - <https://mkdocstrings.github.io/python/usage/configuration/docstrings/>
- **mkdocstrings knobs that matter here:** `filters: ["!^_"]` (private
  hiding — already set), `show_if_no_docstring: true` (renders undocumented
  members — already set; pairs badly with G4 because undocumented stable
  symbols render as bare signatures), `merge_init_into_class`,
  `separate_signature`, `show_signature_annotations` (all already set).
- **Coverage gates:**
  - *Presence/style* — ruff's pydocstyle rules: `select = ["D"]` with
    `[tool.ruff.lint.pydocstyle] convention = "google"`; typically `D1*`
    (missing docstrings) enabled selectively on `src/` with per-file-ignores
    for tests. This repo selects `["E","F","I","UP","B","SIM","C901"]` — no
    `D` at all.
  - *Percentage* — `interrogate --fail-under=N` as a pre-commit/CI gate
    (common floors: 80% overall, 100% on the public facade).
    <https://towardsdatascience.com/automate-your-python-code-documentation-with-pre-commit-hooks-35c7191949a4/>
  - *Style consistency* — `pydocstyle --convention=google` or the ruff
    equivalent; AST-based mixed-style detectors exist (pycmdcheck DC010).
  - Ratchet, don't cliff: measure today (fx1: 74.6% documented;
    quant_fund: 72.3%), set the floor at the measured value, raise only.
    This repo already ratchets coverage (`fail_under = 80`, "raise, never
    lower") and mypy strictness per-module — the same governance shape
    applies to docstrings.
- **Generated reference must cover the product.** mkdocstrings pages exist
  for 5 `quant_fund` areas; zero for `fx1` (G7). The package with the
  semver guarantee should have the most complete rendered reference.

### 2.4 Typed-API ergonomics: Protocol / overload / TypedDict / py.typed

- **Protocol vs ABC.** PEP 544 `Protocol` = structural subtyping: right for
  *boundary contracts* and code you don't control (backends, feature
  pipelines, model interfaces — exactly what `fx1.forecast.protocol` and
  `fx1.serve.backends.InferenceBackend` do). ABC = nominal: right when you
  own the hierarchy, need shared implementation, or need instantiation-time
  enforcement. `@runtime_checkable` enables `isinstance` but checks only
  *attribute presence*, never signatures — document that caveat wherever
  used. `fx1` currently uses Protocol correctly in 2 places and
  `runtime_checkable` nowhere (fine), but registry/factory code
  (`create_model`, `get_backend`) would benefit from runtime-checkable
  validation with an explicit error, since registry inputs come from
  config, not the type checker.
  - PEP 544: <https://peps.python.org/pep-0544/>
  - Real Python, "Implementing Interfaces in Python: ABCs and Protocols":
    <https://realpython.com/python-interface/>
  - Stanza, "Protocols vs Abstract Base Classes" (boundary → Protocol,
    shared impl → ABC, combine strategically):
    <https://www.stanza.dev/courses/python-architecture/protocols/python-architecture-protocols-vs-abc>
- **`TypedDict` for JSON-shaped boundaries.** Receipts, manifests, ledger
  entries, and eval results are `dict[str, …]` at the edges; `TypedDict`
  gives static checkers a schema without runtime cost, and `total=False` /
  `NotRequired` models additive-schema evolution (which is exactly the
  `SFTExample`/receipt "additive fields" promise in
  `FX1_API_STABILITY.md`). Pydantic stays right where runtime validation is
  the point (harness commands, configs) — the repo already uses it there.
  Zero `TypedDict` in `fx1` today (G11).
- **`@overload` for union-dispatch signatures.** Functions like
  `get_backend(name)` returning different concrete types per literal, or
  loaders with `Path | str`, get precise return types via `overload` +
  `Literal`. Zero overloads in `fx1` today.
- **PEP 561 `py.typed`.** Without the marker, type checkers treat the
  package as untyped *even if every line is annotated*. `quant_fund` ships
  `py.typed` containing `partial` (honest about gradual strictness); `fx1`
  ships none (G6). Since both ride the same wheel, add
  `src/fx1/py.typed` (empty = fully typed claim, `partial` = honest) and a
  `[tool.hatch.build.targets.wheel.force-include]` entry mirroring the
  `quant_fund` one.
  - PEP 561: <https://peps.python.org/pep-0561/>

### 2.5 Changelog conventions

- **Keep a Changelog 1.1.0**: `Added / Changed / Deprecated / Removed /
  Fixed / Security` under `## [x.y.z] - YYYY-MM-DD` (ISO 8601), with an
  `## [Unreleased]` section that becomes the release; "yanked" releases
  documented explicitly. Human-first, linkable, and each entry should name
  the symbol/behavior, not the commit. <https://keepachangelog.com/en/1.1.0/>
- This repo complies and goes further by naming the version source
  (`fx1.__version__`). Two hardening opportunities:
  1. **`### Deprecated` discipline** — every deprecation (once §5.3 lands)
     gets a changelog entry in the release that introduces the warning
     *and* the release that removes the symbol (`### Removed`).
  2. **Release/tag test** — assert every `## [x.y.z]` header in
     `CHANGELOG.md` has a matching git tag and equals
     `fx1.__version__` for the latest non-Unreleased entry (cheap, catches
     forgotten bumps).

### 2.6 Docs-as-tests: doctest, `--doctest-modules`, executable examples

- **doctest / pytest integration.** `pytest --doctest-modules` executes
  `>>>` examples in docstrings; `--doctest-glob="*.md"` extends it to prose
  files; make permanent via `addopts`. Default `doctest_optionflags =
  ELLIPSIS`; use `--doctest-report`, `--doctest-continue-on-failure` as
  needed. Deterministic examples only — no wall-clock, no dict ordering
  assumptions, no network (this repo's honesty rules make deterministic
  SYNTHETIC examples natural: label them, per `AGENTS.md` rule 2).
  - <https://docs.pytest.org/en/stable/how-to/doctest.html>
- **Executable example scripts** (what `examples/` already does) are the
  stronger lane for multi-step workflows: each script is linted, type-
  checked (`uv run mypy examples`), and run offline in a subprocess with
  sockets blocked. Best practice: keep both — doctests for API-level
  micro-contracts (one function, one output), scripts for end-to-end
  narratives; cross-link them from the generated reference.
- **Docs drift guards.** `tests/fx1/test_docs_drift.py` (commands, make
  targets, import paths must resolve) is a strong, under-scoped pattern: it
  covers `README.md` + `docs/FX1*.md` (8 files) of 155 docs. Widening the
  glob (at least to `docs/fx1_harness.md`, `docs/examples.md`,
  `examples/README.md`) is nearly free.

### 2.7 Deprecation policy mechanics (PEP 702 + runtime warnings)

- **PEP 702 / `@warnings.deprecated`** (stdlib 3.13+; `typing_extensions`
  ≥ 4.5 backport): marks functions, classes, methods, `TypedDict`/
  `NamedTuple` fields, and individual `overload`s as deprecated. Static
  checkers (pyright, mypy ≥ 1.13 with `enable_error_code =
  deprecated`) flag *usage sites*; at runtime it emits
  `DeprecationWarning` (on call / instantiation / subclassing) unless
  `category=None`. Message lands in `__deprecated__`. Repo requires
  Python ≥ 3.12, so the `typing_extensions` backport is the portable route.
  - PEP 702: <https://peps.python.org/pep-0702/>
  - `warnings.deprecated` docs: <https://docs.python.org/3/library/warnings.html>
- **Cycle discipline.** Widely used policy (numpy, pandas, PyPA packaging
  guide): announce in MINOR with a warning + changelog `### Deprecated` +
  docstring note, remove no earlier than the next MINOR (≥ 1 release of
  overlap); MAJOR bumps may remove anything already deprecated.
  `FX1_API_STABILITY.md` promises "at least one minor version" — consistent;
  the gap is machinery (G3), including the pytest global
  `ignore::DeprecationWarning` which must gain a targeted
  `error::DeprecationWarning:fx1.*` (and `quant_fund.*`) filter so our own
  tests *see* deprecations while third-party noise stays muted.
  - NumPy deprecation policy: <https://numpy.org/neps/nep-0023-backwards-compatibility.html>
  - PyPA Packaging "Deprecation practices":
    <https://packaging.python.org/en/latest/guides/deprecating/>

---

## 3. Audit of the current state (measured 2026-09-28)

### 3.1 `fx1` public surface inventory

Distribution `fx-1` 0.4.0, wheel = `src/fx1` + `src/quant_fund`, console
scripts `fx1 = fx1.cli:app` (+ `quant`, `dipcatcher`, `verify-ledger`,
`mc-engine` from `quant_fund`). 65 modules under `src/fx1`.

| Module | `__all__` | Notes |
|---|---|---|
| `fx1` (root) | **none** | exports `__version__`, `BASE_MODEL` implicitly |
| `fx1.data` | 18 entries | incl. `"sources"` — **broken at runtime (G1)** |
| `fx1.data.sources` | 24 entries | adapter/registry/router facade |
| `fx1.eval` | 21 entries | `compare_runs` undocumented |
| `fx1.train` | 12 entries | `issue_receipt` undocumented |
| `fx1.serve` | 13 entries | `build_manifest` undocumented |
| `fx1.forecast` | 20 entries | `load_harness_config` undocumented; `__all__` sort order broken (`UntrustedArtifactError` before `ModelNotRegistered`) |
| `fx1.bench` | 5 entries | clean |
| `fx1.harness` | **none** | named *stable* in `FX1_API_STABILITY.md` |
| `fx1.honesty` | **none** | named *stable* (`validate_fx1_output`) |
| `fx1.modelcard`, `fx1.mrm`, `fx1.reward`, `fx1.sbom`, `fx1.doctor`, `fx1.hypotheses`, `fx1.cli` | **none** | `fx1.mrm.compile_dossier` named *stable* |

Counts: 134 `__all__` entries across the 8 declaring modules; 358 public
defs/constants package-wide.

### 3.2 The `fx1.data.sources` recursion bug (G1) — reproduced

```python
# fx1/data/__init__.py (current)
def __getattr__(name: str):  # lazy: keep base import light
    if name == "sources":
        from fx1.data import sources   # ← re-enters this __getattr__
        return sources
    raise AttributeError(name)
```

Reproduced on this checkout (default recursion limit):

| Access pattern | Result |
|---|---|
| `from fx1.data import sources` | `RecursionError` |
| `import fx1.data; fx1.data.sources` | `RecursionError` |
| `from fx1.data import *` | `RecursionError` (star-import walks `__all__`, hits `"sources"`) |
| `from fx1.data.sources.registry import REGISTRY` | OK (submodule import sets the attribute directly, `__getattr__` never fires) |

Why: `from fx1.data import sources` inside the handler triggers
`importlib._handle_fromlist`, which probes `hasattr(fx1.data, "sources")`
*before* importing the submodule — re-entering `__getattr__` forever. The
existing test suite misses it because every test imports the deep path
(`fx1.data.sources.adapters` etc.), and `test_docs_drift.py` only resolves
`from fx1.x import y` patterns found in the 8 covered docs (the one doc
hit, `docs/FX1.md`'s `from fx1.data import build_corpus`, doesn't touch
`sources`). Any documentation generator pointed at `fx1.data` (G7's fix)
would crash the docs build — the bug currently *masks itself* by making
`fx1` undocumented.

### 3.3 Docstring coverage (AST scan, module+class level)

| Package | Public defs/consts | Missing docstring | Missing % | `>>>` doctests |
|---|---|---|---|---|
| `fx1` | 358 | 91 | 25.4% | **0** |
| `quant_fund` | 4676 | 1294 | 27.7% | **0** |

Worst `fx1` clusters (full list measured; representative):
`data/sources/adapters.py` (10 — every adapter `describe`/`fetch`),
`forecast/config.py` (5 — all `*Section` config classes + loader),
`train/receipts.py` (4 — incl. `TrainingReceipt`, `issue_receipt`: the
*stable* receipt API), `train/cluster.py`, `train/tracking.py` (Tracker
methods), `eval/contamination.py` (3), `harness.py` (5 — incl.
`HarnessResult`, `list_commands`: the *stable* harness API),
`serve/backends.py` (3 `complete` methods), `forecast/protocol.py`
(`load`/`predict` on the contract Protocols).

`__all__`-promised symbols missing docstrings (6): `fx1.eval.compare_runs`,
`fx1.train.issue_receipt`, `fx1.serve.build_manifest`,
`fx1.forecast.load_harness_config`, `fx1.data.sources.get_spec`,
`fx1.data.sources.list_sources`. Under mkdocstrings
`show_if_no_docstring: true` these render as bare signatures — the docs
build succeeds while the stable surface documents itself as nothing.

Ruff config: `select = ["E","F","I","UP","B","SIM","C901"]` — **no `D`
(pydocstyle) rules**; no interrogate/pydocstyle anywhere; no docstring gate
in CI.

### 3.4 Versioning & distribution state

- Canonical: `fx1.__version__ = "0.4.0"` → hatch dynamic version →
  `fx_1-0.4.0.dist-info` (verified installed). `quant_fund.__version__`
  re-exports it. `CHANGELOG.md` names `fx1.__version__` as the source. ✅
- `fx1 --version` echoes `fx1 {__version__}` and is test-locked. ✅
- **No `py.typed` in `src/fx1`** while `src/quant_fund/py.typed` =
  `partial\n` is force-included into the wheel. fx1 annotations are
  invisible to downstream type checkers (G6). ❌
- **Stale editable install (transient)**: `.venv` was observed holding *two*
  dist-infos from the same directory — `dipcatcher-1.0.0` (the mid-restore
  pyproject variant; entry points `quant` + `dipcatcher` only) and
  `fx_1-0.4.0` (HEAD). Whichever `.pth` wins decides what
  `importlib.metadata.version("fx-1")` reports and which scripts exist.
  A re-check on 2026-09-28 21:58 IST found only `fx_1-0.4.0.dist-info`
  remaining (`version("fx-1") == "0.4.0"`; `version("dipcatcher")` raises
  `PackageNotFoundError`) — the condition self-cleared on re-sync, but
  nothing detects or prevents it (G10). ⚠️
- Model-artifact versioning: documented policy only (no checkpoints
  in-repo, by design). Receipts hash-chain; `fx1.modelcard` exists with
  `save`/`load` (both undocumented). The dual-track design (registry ints +
  `fx-1.vX.Y` semantic tags + content hashes) matches §2.2 sources; what is
  missing is the *alias* concept (`@champion`-style mutable pointer) and an
  explicit statement that artifact stores are write-once — worth one
  paragraph in `FX1_API_STABILITY.md` when the first checkpoint lands.

### 3.5 Docs structure

- `mkdocs.yml`: material theme, mkdocstrings `python` handler with
  `paths: [src]`, `docstring_style: google`, `filters: ["!^_"]`,
  `show_if_no_docstring: true`; `validation.{nav,links}: warn`; CI builds
  `--strict` (warnings = failures). ✅ config; ❌ coverage:
- `docs/api/` = 5 pages + index, 19 `:::` targets, **all `quant_fund.*`**;
  `docs/api/index.md` says so explicitly ("Generated from the docstrings in
  `src/quant_fund`"). **Zero `fx1` pages** (G7) — no nav section, no `:::`
  target, and `signature_crossrefs: false` means even prose references
  don't link.
- Nav hygiene: 155 `.md` under `docs/` at audit time (156 once this file
  landed); 133 nav-referenced; 7 in
  `exclude_docs` (honesty-motivated: Sharpe/P&L-headlining docs); **16
  omitted** — the entire `docs/SOTA/` lane (03,04,06,07,08,09,10,11,13,14,
  15,16,17,18,19,20) — each producing an `omitted_files: warn` on every strict
  build (G9). No nav entry points at a missing file. ✅
- `docs/examples.md` is a snippet-include of `examples/README.md`
  (`--8<--`) — single-source, good pattern.
- `tests/fx1/test_docs_drift.py` covers `README.md` + `docs/FX1*.md` (8
  files): fx1 CLI commands, make targets, import paths. 147 docs unchecked.

### 3.6 Deprecation & warning machinery

- `DeprecationWarning` / `warnings.warn` / `@deprecated` occurrences in
  `src/fx1` + `src/quant_fund`: **0** (G3). The policy in
  `FX1_API_STABILITY.md` ("the old one emitting a `DeprecationWarning` for
  at least one minor version") has never yet been exercised and has no
  helper, no test, and no changelog `### Deprecated` entry precedent.
- `[tool.pytest.ini_options] filterwarnings = ["ignore::DeprecationWarning"]`
  globally silences them — so if a deprecation *were* added tomorrow, the
  suite would not notice it firing (or misfiring). A targeted
  `error::DeprecationWarning:(fx1|quant_fund)\..*` entry (first-match-wins
  ordering: put it *before* the ignore) is required for the policy to be
  testable.

### 3.7 Typed-API ergonomics in `fx1`

| Construct | Count (65 modules) |
|---|---|
| `Protocol` (definitions/uses) | 6 occurrences; 2 contract Protocols (`serve.backends.InferenceBackend`, `forecast.protocol.{ForecastModel, FeaturePipeline, Fx1Model}`) |
| `TypedDict` | **0** |
| `@overload` | **0** |
| `@runtime_checkable` | **0** |
| `py.typed` | **absent** |

Pydantic is used well at runtime-validation boundaries (`harness.py`
commands, train/forecast configs). The gaps are at *typing* boundaries:
receipt/manifest/ledger JSON shapes flow as plain dicts, backend factories
dispatch on strings without `Literal`/overloads, and the structural
contracts are not runtime-checkable where registry data (config-driven,
untyped) enters.

### 3.8 Changelog & examples

- `CHANGELOG.md`: Keep a Changelog 1.1.0 + SemVer declared; `[Unreleased]`
  active; entries name symbols and behavior changes precisely (e.g. the
  PSR unit-safety fix documents the ~15.9× diagnostic shift "by design").
  No `### Deprecated`/`### Removed` sections have ever been needed yet. ✅
- `examples/`: 5 scripts, jupytext percent-format, offline subprocess
  runner with socket blocking + banned-import prefixes
  (`quant_fund.paper/execution/api/backtest` — order-execution surfaces are
  kept out of examples, consistent with the no-live-trading rule). `make
  examples` = ruff + format + mypy + pytest. This is the repo's strongest
  docs-as-tests asset; it covers `quant_fund` workflows — **no example
  exercises the `fx1` public API** (corpus build, eval suite, honesty
  validation would all run offline and SYNTHETIC-labeled).

---

## 4. Consistency issues (summary table)

| Where | Inconsistency |
|---|---|
| `fx1.data.__all__` | promises `"sources"`; attribute access raises `RecursionError` (§3.2) |
| `FX1_API_STABILITY.md` vs `fx1.harness/honesty/mrm` | symbols declared *stable* live in modules with no `__all__` — boundary is prose-only |
| `py.typed` | present (`partial`) for `quant_fund`, absent for `fx1`, same wheel |
| `docs/api/` | generated reference covers the harness (`quant_fund`), not the product (`fx1`) |
| pytest `filterwarnings` | global `ignore::DeprecationWarning` contradicts the deprecation policy's testability |
| `fx1.forecast.__all__` | not sorted (`UntrustedArtifactError` precedes `ModelNotRegistered`); ruff `RUF022` would catch — `RUF` not in `select` |
| `.venv` | two editable dist-infos (`dipcatcher 1.0.0` stale + `fx-1 0.4.0`) from one directory — transient; self-cleared on re-sync, still undetected (G10) |
| mkdocs nav | 16 `docs/SOTA/*.md` omitted (incl. this doc) → 16 strict-build warnings |
| doctests | 0 in 735 modules; `addopts` lacks `--doctest-modules` |

---

## 5. Adoption plan

Sequenced by leverage/effort. Every item is additive; none weakens the
honesty contract or existing gates. Items marked **[bugfix]** should land
immediately and directly on `main` per `AGENTS.md`.

### 5.1 [bugfix] Fix the lazy `sources` loader (G1)

Replace the recursing handler in `src/fx1/data/__init__.py` with the
import-module + cache pattern (same shape `quant_fund.__init__` already
uses):

```python
def __getattr__(name: str):  # lazy: keep base import light
    if name == "sources":
        import importlib

        module = importlib.import_module("fx1.data.sources")
        globals()["sources"] = module  # cache: second access skips __getattr__
        return module
    raise AttributeError(name)
```

Regression test (belongs in `tests/fx1/test_data_sources.py` or a new
`test_public_surface.py`), run in a **subprocess** so module-cache state
can't mask it:

```python
def test_lazy_sources_attribute_resolves() -> None:
    script = (
        "import fx1.data as d\n"
        "assert d.sources.REGISTRY is not None\n"
        "assert d.sources is d.sources  # cached, no recursion\n"
        "from fx1.data import sources as s2\n"
        "assert s2 is d.sources\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], check=False,
                               capture_output=True, text=True, cwd=_ROOT)
    assert completed.returncode == 0, completed.stderr
```

Also add `"sources"` resolution to the drift test's import-path checks.
If a generic helper is wanted for future lazy submodules, extract
`fx1._lazy.submodule_getattr(__name__, name, mapping)` — but with one call
site, inline is fine.

### 5.2 API-surface snapshot + griffe gate for `fx1` (G2)

1. **Snapshot test** mirroring `tests/unit/test_public_api.py`:
   `tests/fx1/test_api_snapshot.py` renders, for each `fx1` module with
   `__all__` (and, after §5.8, for the newly-declaring stable modules), the
   sorted `__all__` plus `inspect.signature` of every exported callable and
   field list of every exported model/dataclass into
   `tests/fx1/fx1_api_snapshot.txt`; assert equality. Intentional changes
   update the snapshot in the same commit — the diff *is* the API review.
   Reuse `render_public_api`'s helpers (move them to a shared
   `tests/_api_render.py` or duplicate minimally).
2. **Stable-surface lock**: a second, small snapshot of exactly the symbols
   `FX1_API_STABILITY.md` promises (Harness.run/list_commands,
   build_corpus/build_full_corpus, run_suite/EvalTask,
   validate_fx1_output, issue_receipt/verify_training_receipt,
   sign_release/verify_release, compile_dossier) with a comment pointing at
   the doc — making the policy machine-checked. Assert additionally that
   the doc's table and the snapshot list are in sync (parse the markdown
   table's backticked paths like `test_docs_drift.py` already parses docs).
3. **Griffe breaking-change check in CI** (`fx1.yml`, needs
   `fetch-depth: 0`):

```yaml
- name: API breakage check (fx1)
  run: uvx griffe check --search src --format github fx1
```

   Default compares against the latest tag; during 0.x (semver permits
   breaking minors) run it **advisory** (`continue-on-error: true` + PR
   annotation) or `--against origin/main` on PRs, and make it blocking at
   1.0. Optionally extend to `quant_fund` in `ci.yml`.
4. **Version coherence test**: `importlib.metadata.version("fx-1") ==
   fx1.__version__` (fails loudly on the stale-install condition of G10) —
   plus `make sync` guidance: `uv sync --frozen --reinstall-package fx-1`
   after a name/version change.

### 5.3 Deprecation helper + warning policy (G3)

1. Add `src/fx1/_compat.py` (private; not API):

```python
"""Deprecation helpers — PEP 702 backport shim (repo floor is Python 3.12)."""
from __future__ import annotations

import warnings
from typing import TypeVar

try:  # Python >= 3.13
    from warnings import deprecated
except ImportError:  # pragma: no cover
    from typing_extensions import deprecated  # type: ignore[assignment]

_F = TypeVar("_F")


def deprecate(obj: _F, message: str, *, removal: str) -> _F:
    """Wrap `obj` so use emits DeprecationWarning; record removal version.

    `removal` is the planned fx1 version (e.g. "0.6.0") per
    docs/FX1_API_STABILITY.md — at least one minor version of overlap.
    """
    return deprecated(f"{message} (removal planned in fx1 {removal})")(obj)  # type: ignore[return-value]
```

   (`typing_extensions` is already a transitive dep of mypy/pydantic; if it
   must be explicit, add it to `[project] dependencies` — it is tiny and
   universally vendored.)
2. **Pytest filter** (in `[tool.pytest.ini_options]`, first-match-wins, so
   these go *above* the blanket ignore):

```toml
filterwarnings = [
    "error::DeprecationWarning:fx1.*",
    "error::DeprecationWarning:quant_fund.*",
    "ignore::DeprecationWarning",
]
```

   Own-code deprecations become test failures unless a test explicitly
   exercises them with `pytest.warns(DeprecationWarning)`; third-party
   noise stays ignored.
3. **Policy test**: `tests/fx1/test_deprecation_policy.py` asserts every
   object with a `__deprecated__` attribute (walk the `__all__`s) is (a)
   listed under `### Deprecated` in `CHANGELOG.md`, and (b) carries a
   `removal planned in fx1 X` suffix with `X > __version__`. This makes the
   changelog and the code unable to drift.
4. **mypy**: `enable_error_code = ["deprecated"]` (mypy ≥ 1.13 supports
   PEP 702) so internal use of deprecated symbols is flagged at typecheck.
5. Update `FX1_API_STABILITY.md` §Deprecation to name the helper and the
   exact cycle: announce in MINOR (warning + `### Deprecated` + docstring
   `Deprecated:` note), remove no earlier than the *next* MINOR, log under
   `### Removed`.

### 5.4 Docstring gate (G4) — ratchet, not cliff

1. **Enable ruff `D` (pydocstyle, google convention) narrowly**:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "C901", "D", "RUF022"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["B018", "D"]
"examples/**" = ["D"]
```

   Start with the *summary* rules only (`D1` family = missing docstrings,
   `D200/D400/D415` = one-liner hygiene) and **baseline-exempt the existing
   1385 gaps** via a checked-in ignore list or by gating only the public
   surface: a small `scripts/check_public_docstrings.py` (AST walk of every
   `__all__` export + the `FX1_API_STABILITY.md` stable list) that fails on
   any missing/empty docstring. That is 6 symbols today (§3.3) — fix them
   in the enabling PR, then ratchet: the script's floor is "zero missing on
   `__all__`", forever.
2. **Percentage ratchet for the whole tree** (optional second stage):
   `interrogate -c pyproject.toml` with
   `fail-under = 74` for `src/fx1` and `72` for `src/quant_fund` (today's
   measured values), in `make fx1-gate` / `make lint`; raise, never lower —
   same governance as `fail_under = 80` coverage.
3. Priority order for backfilling the 91 `fx1` gaps: stable-surface modules
   first (`harness.py`, `train/receipts.py`, `serve/signing.py`,
   `data/corpus.py`, `eval/suite.py`, `honesty.py`, `mrm.py`), then
   `__all__` exports, then everything else. Each docstring: one-line
   summary (imperative, ≤ 100 cols per ruff `line-length`), Google
   `Args:/Returns:/Raises:` where non-obvious, and a `Deprecated:` note
   when §5.3 applies.

### 5.5 Doctests (G5)

1. Add to `addopts`: `--doctest-modules` scoped by a second pytest lane —
   do **not** add it to the default gate globally (735 modules × collection
   cost, and `quant_fund` imports are heavy). Cleanest: a make target +
   CI step:

```make
doctest: ## Run docstring examples for the public surface
	PYTHONPATH=src uv run pytest --doctest-modules src/fx1 -q -p no:cacheprovider
```

2. Seed doctests where they are deterministic and offline: purity/honesty
   validators (`validate_fx1_output` on a labeled SYNTHETIC example),
   schema validators (`validate_forecast_schema`), ledger hashing
   (`compute_hash` on a fixed input), `classify_need` routing, dip-event
   detection on a fixed synthetic frame. Target: every `__all__`-exported
   *function* in the stable list has ≥ 1 doctest within one minor version.
   Mark any example data SYNTHETIC in the doctest text itself (honesty rule
   2 applies to docstrings too).
3. `--doctest-glob="*.md"` on `docs/FX1*.md` is a candidate but risky
   (prose backticks aren't valid doctests); prefer extending
   `test_docs_drift.py` instead (§5.9).

### 5.6 `py.typed` for `fx1` (G6)

Create `src/fx1/py.typed` containing `partial` (honest: mypy runs on `fx1`
with the global non-strict tier), and add to pyproject:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/fx1/py.typed" = "fx1/py.typed"
```

Test alongside the existing `test_py_typed_marks_the_package_partial`:
assert both markers exist and ship in the built wheel
(`uv build --wheel` + zipfile listing in CI, or a static
`force-include` config assertion mirroring
`test_mypy_strict_flags_are_per_module`'s tomllib approach). Flip `partial`
→ empty only when the `fx1` modules reach the strict mypy tier.

### 5.7 mkdocstrings pages for `fx1` (G7)

New `docs/api/fx1/` tree, one page per subpackage, added to nav under
**API reference**:

```markdown
<!-- docs/api/fx1/index.md -->
# fx-1 API reference

Generated from the docstrings in `src/fx1`. Versioned per
`fx1.__version__` (see [API stability](../../FX1_API_STABILITY.md)).
Corpus/eval/train surfaces are research infrastructure; nothing here
executes trades.

::: fx1.data
::: fx1.eval
::: fx1.train
::: fx1.serve
::: fx1.forecast
::: fx1.bench
::: fx1.harness
::: fx1.honesty
```

(Split into per-module pages like the `quant_fund` set if one page gets
long.) Prerequisites: §5.1 (mkdocstrings walks `__all__` → the recursion
bug would break the strict build) and ideally the §5.4 stable-surface
docstring pass, because `show_if_no_docstring: true` will otherwise
publish bare signatures for `issue_receipt` and friends. Add
`docs/api/index.md` links + nav entries in the same PR.

### 5.8 `__all__` for the stable modules (G8)

Add explicit `__all__` to the 9 modules named or implied by
`FX1_API_STABILITY.md` and the CLI: `fx1` root (`["__version__",
"BASE_MODEL"]`), `fx1.harness` (`Harness`, `HarnessRole`, `HarnessCommand`,
`HarnessResult`), `fx1.honesty` (`validate_fx1_output`,
`Fx1HonestyError`, `FORBIDDEN_HEADLINE_TOKENS`), `fx1.mrm`
(`compile_dossier`, `DossierSection`), `fx1.modelcard` (`ModelCard`),
`fx1.reward`, `fx1.sbom`, `fx1.doctor`, `fx1.hypotheses`. Then the §5.2
snapshot covers them and `griffe check` treats non-`__all__` names as
private (fewer false breakages). Enable `RUF022` (sorted `__all__`) to fix
`fx1.forecast` ordering automatically.

### 5.9 Docs hygiene (G9 + drift widening)

1. Add a **SOTA** nav section (or `exclude_docs` entries with a comment if
   the lane is meant to stay internal — but these docs are citable research
   notes, and publishing them matches the `docs/SOTA_CANON_ROADMAP`'s
   intent). Nav addition kills all 16 `omitted_files` warnings:

```yaml
  - SOTA:
      - CI/CD: SOTA/16-ci-cd.md
      - Code quality: SOTA/19-code-quality.md
      - API & docs: SOTA/17-api-docs.md
      # ... remaining lanes
```

2. Widen `test_docs_drift.py::_DOC_PATHS` to include
   `docs/fx1_harness.md`, `docs/examples.md`, `examples/README.md`, and
   `docs/api/fx1/*.md` (the new pages should also pass the
   commands/targets/imports checks). Cheap: the machinery exists.
3. Add a nav-completeness test (or rely on `--strict` once SOTA pages are
   in nav): assert `set(docs/**/*.md) - excluded ⊆ nav` so future omissions
   fail CI instead of warning.

### 5.10 Environment coherence (G10)

- `make sync` note in `AGENTS.md`-adjacent docs: after any `[project]`
  name/version change, run `uv sync --frozen --reinstall-package fx-1` (or
  `--reinstall`) so a stale dist-info (the `dipcatcher-1.0.0` variant
  observed mid-restore) cannot shadow `fx-1`. The condition self-cleared on
  a later re-sync, which is exactly why it needs a detector: the §5.2.4
  `importlib.metadata.version("fx-1") == fx1.__version__` test turns this
  from tribal knowledge into a gate.

### 5.11 Typed-API ergonomics pass (G11)

Incremental, per-module, no big-bang:

1. `@runtime_checkable` on `InferenceBackend` + `ForecastModel`/
   `FeaturePipeline`, with a validating error in `get_backend`/
   `create_model` (registry input is config-driven; fail-closed matches
   ADR-0004). Document the presence-only caveat in the Protocol
   docstrings.
2. `TypedDict` for the JSON-at-the-boundary shapes: `TrainingReceipt`
   payload, release manifest (`build_manifest` return), ledger entry dicts,
   eval result blobs. These become the schema the "additive fields only"
   promise in `FX1_API_STABILITY.md` refers to — `NotRequired` marks
   additive-optional fields explicitly.
3. `Literal` + `@overload` for factory dispatch
   (`get_backend("hosted_k3") -> HostedK3Backend` etc.) once (1) lands.
4. Keep pydantic where runtime validation is the point (harness, configs);
   dataclasses for internal value objects; don't mix the three without
   reason.

### 5.12 `fx1` example script (extends §3.8)

Add `examples/06_fx1_corpus_and_honesty.py`: offline, SYNTHETIC-labeled —
build a tiny corpus from a temp receipt, run `validate_fx1_output` on a
compliant and a violating output (show fail-closed), verify a training
receipt round-trip. Wire into `tests/examples/test_examples_run.py`'s
`EXAMPLES` tuple (the socket-blocking runner already enforces offline).
This is the first executable doc of the `fx1` public API and doubles as an
integration smoke of §5.1's fixed import path.

### Sequencing

| PR | Contents | Depends on |
|---|---|---|
| 1 **[bugfix]** | §5.1 recursion fix + subprocess regression test | — |
| 2 | §5.8 `__all__` on stable modules + `RUF022` | 1 |
| 3 | §5.2 snapshot + stable-surface lock + version coherence test | 2 |
| 4 | §5.4 public-surface docstring gate (fix the 6 `__all__` gaps) | 2 |
| 5 | §5.3 deprecation helper + pytest filters + policy test | — |
| 6 | §5.6 `py.typed` for fx1 | — |
| 7 | §5.7 mkdocstrings fx1 pages + nav (+ §5.9 SOTA nav) | 1, 4 |
| 8 | §5.5 doctest lane + seed doctests | 4 |
| 9 | §5.11 typed-ergonomics pass (incremental) | 3 |
| 10 | §5.12 fx1 example script | 1 |
| CI | griffe advisory check (§5.2.3) in `fx1.yml` | 3 |

---

## 6. Sources

**API design & boundaries**
- PEP 8 — naming/underscore conventions: <https://peps.python.org/pep-0008/>
- PEP 562 — module `__getattr__`/`__dir__`: <https://peps.python.org/pep-0562/>
- Griffe — API breaking-change checking in CI: <https://mkdocstrings.github.io/griffe/guide/users/checking/>, <https://github.com/mkdocstrings/griffe>
- PyPA — API guides & deprecation practices: <https://packaging.python.org/en/latest/guides/deprecating/>

**Versioning (packages + ML artifacts)**
- SemVer 2.0.0: <https://semver.org/spec/v2.0.0.html>
- Keep a Changelog 1.1.0: <https://keepachangelog.com/en/1.1.0/>
- MLflow Model Registry (versions, aliases, immutability): <https://mlflow.org/docs/latest/ml/model-registry/>
- MLflow registry checklist (monotonic ints vs semver families, write-once): <https://mlflow.org/articles/ai-model-registry-management-checklist/>
- MLflow shared-registry governance: <https://mlflow.org/articles/role-of-shared-model-registry/>
- Atlan AI-model versioning (2026): <https://atlan.com/know/ai-model-versioning-best-practices/>
- Introl model versioning infrastructure (2025): <https://introl.com/blog/model-versioning-infrastructure-mlops-artifact-management-guide-2025>

**Documentation standards**
- Griffe docstring parsers & style recommendation: <https://mkdocstrings.github.io/griffe/docstrings/>, <https://mkdocstrings.github.io/griffe/guide/users/recommendations/docstrings/>
- mkdocstrings-python docstring options (`docstring_style`, `show_if_no_docstring`, `filters`): <https://mkdocstrings.github.io/python/usage/configuration/docstrings/>
- numpydoc standard: <https://numpydoc.readthedocs.io/>
- Ruff pydocstyle (`D`) rules: <https://docs.astral.sh/ruff/rules/#pydocstyle-d>
- interrogate (docstring-coverage gate): <https://github.com/econchick/interrogate>; pre-commit walkthrough: <https://towardsdatascience.com/automate-your-python-code-documentation-with-pre-commit-hooks-35c7191949a4/>
- MkDocs validation / `exclude_docs`: <https://www.mkdocs.org/user-guide/configuration/#validation>

**Typed-API ergonomics**
- PEP 544 — Protocols / structural subtyping: <https://peps.python.org/pep-0544/>
- PEP 561 — `py.typed`: <https://peps.python.org/pep-0561/>
- PEP 589 — TypedDict: <https://peps.python.org/pep-0589/>
- PEP 702 — `@warnings.deprecated`: <https://peps.python.org/pep-0702/>; stdlib docs: <https://docs.python.org/3/library/warnings.html>
- Real Python — ABCs vs Protocols: <https://realpython.com/python-interface/>
- Stanza — Protocols vs ABC decision guide: <https://www.stanza.dev/courses/python-architecture/protocols/python-architecture-protocols-vs-abc>
- NumPy backwards-compatibility/deprecation policy (NEP 23): <https://numpy.org/neps/nep-0023-backwards-compatibility.html>

**Docs-as-tests**
- pytest doctest integration (`--doctest-modules`, `--doctest-glob`, `doctest_optionflags`): <https://docs.pytest.org/en/stable/how-to/doctest.html>
- stdlib doctest: <https://docs.python.org/3/library/doctest.html>
- jupytext percent-format scripts (examples lane): <https://jupytext.readthedocs.io/>

---

*Research-infrastructure note. Contains no performance claim, no promotion,
and no live-trading authorization. SYNTHETIC references herein are
correctness illustrations, not market evidence.*
