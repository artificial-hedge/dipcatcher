# Code quantity and quality inventory

Line count is a size signal, not a quality claim. In particular, generated
files, copied dependencies, comments, blank lines, and repeated padding can
make a repository larger without adding a tested feature. The repository
therefore does not label all physical lines as “quality LOC.”

Run the reproducible inventory with:

```bash
make code-inventory
```

The report counts only Git-tracked Python under `src/`, `tests/`, and
`scripts/`. It publishes physical lines, tokenizer-backed semantic source
lines, comments, docstrings, functions, classes, and test functions. Ignored,
vendored, dependency, build, and generated paths are excluded. A nonzero
minimum turns the report into a fail-closed size check:

```bash
uv run python scripts/code_quality_inventory.py --summary-only --minimum 2400000
```

That command must not be represented as passing until the measured source
actually reaches the threshold. New code should be driven by product and
research requirements and pass the normal lint, type, test, honesty, and
receipt gates; millions of filler lines are not an acceptable substitute.

For longitudinal reporting, use `--output PATH`. The JSON schema is versioned
as `code-quality-inventory/v1`, and the default output includes per-file
records so changes can be reviewed rather than inferred from a single total.
