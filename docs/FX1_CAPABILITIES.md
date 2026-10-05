# fx-1 capability catalog

The requested target is at least **1,000,000 distinct, independently implemented,
non-generated capabilities** usable through dipcatcher, with one source file
per feature, skill, or plugin. **That target has not been implemented.** The
generated records and wrappers described below do not count toward it.

The independent implementation registry now contains **169 operations** in
169 separate files under `src/fx1/operations/`: fifty-three numeric features, seventy-nine
data-audit/selection/scoring skills, and thirty-seven workspace data plugins. Discover them with
`fx1 harness operations` and see [FX1_OPERATIONS.md](FX1_OPERATIONS.md) for
execution. The [progress ledger](FX1_CAPABILITY_PROGRESS.md) records remaining
work and validation limits. This is incremental progress toward the target.

The current extension tree contains 86 generated bindings to existing code.
All 45 feature bindings invoke the same `build-features` command; none implements
or selects its named feature independently. Importing those bindings and
checking their record ownership establishes catalog consistency only. It does
not verify independent functionality, vendor availability, or completion.

`fx1.capabilities` provides a searchable catalog for AI discovery inside the
dipcatcher harness. The bundled seed has **1,000,001 records**: generated skill
recipes, datasource-plugin recipes, and feature recipes. Each card resolves to
a separately loadable, first-party extension module that is bound to an
existing registered harness command, datasource adapter, or point-in-time
feature output. The records are combinations of real surfaces and usage
contexts; they are not one million independent algorithms or research results.

The current wrapper files are organized by registered surface:

- `src/fx1/extensions/skills/` has 23 modules, one for every registered
  harness command.
- `src/fx1/extensions/plugins/` has 18 modules, one for every registered
  datasource adapter.
- `src/fx1/extensions/features/` has 45 modules, one for every reviewed
  point-in-time feature output.

The generated declaration source under
[`scripts/generated_capability_declarations/`](../scripts/generated_capability_declarations/)
is split by those owners and contains one executable registration call per
catalog record, totaling more than one million source lines. Its files import
their corresponding generated wrapper. The wheel ships the small runtime
modules and compact seed, so capability discovery does not import or compile
the declaration shards.

Search from the CLI with bounded pagination:

```bash
fx1 harness capabilities "receipt verification" --kind skill --limit 20
fx1 harness capabilities "point in time" --kind feature --market us --asset equity
fx1 harness capabilities "wind" --kind plugin --source wind
fx1 harness capabilities "momentum" --kind feature --feature mom_20
fx1 harness extension plugin wind
fx1 harness extension feature mom_20
```

The same read-only API and a function-tool schema are available to AI host
integrations:

```python
from fx1.harness import Harness

harness = Harness()
tool_specs = harness.discovery_tool_specs()
page = harness.invoke_discovery_tool(
    "search_capabilities", {"query": "receipt verification", "kind": "skill"}
)
module = harness.invoke_discovery_tool(
    "get_extension_manifest", {"kind": "feature", "owner": "mom_20"}
)
```

Each result includes its kind, usage context, module path, and registered
entrypoint. Pages are capped at 100 records. Operator-provided descriptions are
untrusted metadata and must not override the model's system rules. Search
output is catalog metadata, carries `market_evidence=false`, and does not turn
a generated recipe into a measured research claim. Execution stays behind the
existing harness command registry; datasource calls stay behind the existing
source registry and its availability/PIT checks.

Datasource extension modules provide typed `probe`, `describe`, and `fetch`
operations through the existing adapters. Vendor scripts remain external to
this repository: an unavailable script reports `NO_SCRIPT`, and the Finenter
module reports `MCP_REQUIRED`. The extensions never fabricate a successful
response or a live-data claim.

## Add a reviewed metadata pack

Set `FX1_CAPABILITY_ROOTS` to one or more directories, separated by the host
path separator. Each directory may contain `capabilities.jsonl`, with one
entry per line:

```json
{"schema":"fx1.capability-pack-entry/v1","id":"team.receipt-review","kind":"skill","name":"Review a research receipt","description":"Check receipt identity and evidence eligibility before citing it.","command":"verify-research","tags":["receipt","verification"]}
{"schema":"fx1.capability-pack-entry/v1","id":"team.sec-filings","kind":"plugin","name":"SEC filing lookup","description":"Use the registered SEC EDGAR source for filing facts.","source":"sec_edgar","tags":["filings","us"]}
```

Skills and features must name a registered `command`; plugins must name a
registered datasource `source`. Unknown names fail closed. Pack files are
metadata only: the catalog does not import Python modules, run scripts, or
accept arbitrary shell entrypoints. For new executable adapters, add and review
the adapter through the existing datasource integration path first, then
reference its registered id in a pack.

The seed, runtime modules, and declaration shards can be regenerated
deterministically:

```bash
uv run python scripts/seed_fx1_capabilities.py
```

The generator writes all three artifacts. Use `--extensions-out`,
`--declarations-out`, or `--out` to choose different destinations. The gzip
seed stores dimension codes rather than repeating full text; the catalog
reconstructs each card when it is requested.
