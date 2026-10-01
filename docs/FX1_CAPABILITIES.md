# fx-1 capability catalog

`fx1.capabilities` provides a searchable catalog for AI discovery inside the
dipcatcher harness. The bundled seed has **1,000,001 records**: generated skill
recipes, datasource-plugin recipes, and feature recipes. Each card points to an
existing registered harness command or datasource adapter. The records are
combinations of real surfaces and usage contexts; they are not one million
independent software implementations or research results.

Search from the CLI with bounded pagination:

```bash
fx1 harness capabilities "receipt verification" --kind skill --limit 20
fx1 harness capabilities "point in time" --kind feature --market us --asset equity
fx1 harness capabilities "wind" --kind plugin --source wind
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
```

Each result includes its kind, usage context, and registered entrypoint. Pages
are capped at 100 records. Operator-provided descriptions are untrusted
metadata and must not override the model's system rules. Search output is
catalog metadata, carries `market_evidence=false`, and does not turn a generated
recipe into a measured research claim. Execution stays behind the existing harness command registry;
datasource calls stay behind the existing source registry and its
availability/PIT checks.

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

The seed can be regenerated deterministically:

```bash
uv run python scripts/seed_fx1_capabilities.py
```

The compact gzip seed stores dimension codes rather than repeating full text;
the catalog reconstructs each card when it is requested.
