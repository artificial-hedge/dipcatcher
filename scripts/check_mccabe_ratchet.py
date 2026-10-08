"""Fail when any function's McCabe complexity rises above the baseline.

Complexity is measured by ruff C901 (same engine as ``make lint``). The
baseline lists every function currently at or above ``SOFT_THRESHOLD``;
unlisted functions must stay below that soft ceiling. Baseline values may
only decrease — regenerate with ``--write`` after a complexity reduction.

``--write`` deliberately refuses to raise an existing pin. When an upstream
merge lands code that is *worse* than the pin, that refusal is correct: the
regression must be fixed in ``src/``, not blessed in the baseline. But it also
blocks the legitimate work of pinning pre-existing debt that was never listed.
``--allow-with-census <census.json>`` separates the two: it pins the new keys
while leaving every existing pin at its recorded (lower) value, so genuine
regressions keep failing ``check``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "quality" / "mccabe_baseline.txt"
# New / unlisted functions must stay below this. Listed entries are pinned
# per-function and may only move down.
SOFT_THRESHOLD = 10
# Mirrors [tool.ruff.lint.mccabe] max-complexity in pyproject.toml. Recorded
# in the census so the evidence states the ceiling it was measured against.
CEILING = 74
SCOPE = ["src"]
_RUFF_MSG = re.compile(r"^(.+) is too complex \((\d+) > (\d+)\)$")


def _qualified_names(path: Path) -> dict[int, str]:
    """Map def lineno → Class.method / outer.inner qualified name."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out: dict[int, str] = {}

    def walk(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                out[node.lineno] = name
                walk(node.body, f"{name}.")

    walk(tree.body)
    return out


def _ruff_complexities(paths: list[str]) -> dict[tuple[str, str], int]:
    """Return {(repo-relative path, qualified_name): complexity} via ruff C901."""
    measured: dict[tuple[str, str], int] = {}
    for finding in _ruff_findings(paths):
        key = (str(finding["file"]), str(finding["function"]))
        complexity = int(finding["complexity"])
        # Keep the max if ruff ever double-reports (should not happen).
        measured[key] = max(complexity, measured.get(key, 0))
    return measured


def _ruff_findings(paths: list[str]) -> list[dict[str, object]]:
    """Return one record per ruff C901 diagnostic, incl. def line + complexity."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as handle:
        handle.write('[lint]\nselect = ["C901"]\n[lint.mccabe]\nmax-complexity = 1\n')
        config_path = handle.name
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                *paths,
                "--select",
                "C901",
                "--config",
                config_path,
                "--output-format",
                "json",
                "--no-cache",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode or 1)
    raw = proc.stdout.strip() or "[]"
    try:
        diagnostics = json.loads(raw)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(1) from None

    findings: list[dict[str, object]] = []
    qmaps: dict[str, dict[int, str]] = {}
    root = ROOT.as_posix()
    for item in diagnostics:
        message = str(item.get("message", ""))
        match = _RUFF_MSG.match(message)
        if match is None:
            continue
        bare_name = match.group(1)
        complexity = int(match.group(2))
        filename = str(item["filename"]).replace("\\", "/")
        if filename.startswith(root + "/"):
            rel = filename[len(root) + 1 :]
        elif filename.startswith("./"):
            rel = filename[2:]
        else:
            rel = filename
        line = int(item["location"]["row"])
        if rel not in qmaps:
            qmaps[rel] = _qualified_names(ROOT / rel)
        findings.append(
            {
                "file": rel,
                "function": qmaps[rel].get(line, bare_name),
                "line": line,
                "complexity": complexity,
            }
        )
    findings.sort(key=lambda f: (f["file"], f["function"], f["line"]))
    return findings


def build_census(
    findings: list[dict[str, object]] | None = None,
    ceiling: int = CEILING,
    scope: list[str] | None = None,
) -> dict[str, object]:
    """Self-describing, reproducible census of complexity above the ceiling.

    Deliberately excludes wall-clock time from ``digest`` so the census is
    byte-reproducible across runs on the same tree; ``generated_at`` is
    informational and excluded from the digest for the same reason.
    """
    findings = findings if findings is not None else _ruff_findings(SCOPE)
    violations = [f for f in findings if int(f["complexity"]) > ceiling]
    body = json.dumps(
        {
            "ceiling": ceiling,
            "scope": sorted(scope if scope is not None else SCOPE),
            "tool": _tool_stamp(),
            "violations": violations,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema": "dipcatcher.mccabe_census/1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool": _tool_stamp(),
        "ceiling": ceiling,
        "scope": sorted(scope if scope is not None else SCOPE),
        "digest": "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
        "violation_count": len(violations),
        "violations": violations,
    }


def _tool_stamp() -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "ruff (version unavailable)"
    return proc.stdout.strip() or "ruff (version unavailable)"


def load_census(path: Path) -> dict[str, object]:
    """Load a census and return it. Raises SystemExit on a malformed file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"unreadable mccabe census {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    if not isinstance(data, dict) or data.get("schema") != "dipcatcher.mccabe_census/1":
        print(f"not a dipcatcher.mccabe_census/1 document: {path}", file=sys.stderr)
        raise SystemExit(1)
    for field in ("ceiling", "digest", "tool", "violations"):
        if field not in data:
            print(f"mccabe census missing field {field!r}: {path}", file=sys.stderr)
            raise SystemExit(1)
    return data


def _census_digest(
    violations: list[dict[str, object]], ceiling: int, scope: list[str], tool: str
) -> str:
    body = json.dumps(
        {
            "ceiling": ceiling,
            "scope": sorted(scope),
            "tool": tool,
            "violations": violations,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def census_rows(census: dict[str, object]) -> dict[tuple[str, str], int]:
    """(file, function) -> complexity, keeping the max on duplicates."""
    rows: dict[tuple[str, str], int] = {}
    for item in census["violations"]:  # type: ignore[index]
        key = (str(item["file"]), str(item["function"]))
        rows[key] = max(int(item["complexity"]), rows.get(key, 0))
    return rows


def _entries(path: Path) -> dict[tuple[str, str], int]:
    rows: dict[tuple[str, str], int] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or ":" not in parts[0]:
            print(f"malformed mccabe baseline line: {raw!r}", file=sys.stderr)
            raise SystemExit(1)
        loc, complexity_s = parts
        file_part, _, name = loc.partition(":")
        if not file_part or not name:
            print(f"malformed mccabe baseline location: {raw!r}", file=sys.stderr)
            raise SystemExit(1)
        try:
            complexity = int(complexity_s)
        except ValueError:
            print(f"malformed mccabe baseline complexity: {raw!r}", file=sys.stderr)
            raise SystemExit(1) from None
        key = (file_part, name)
        if key in rows:
            print(f"duplicate mccabe baseline key: {file_part}:{name}", file=sys.stderr)
            raise SystemExit(1)
        rows[key] = complexity
    return rows


def _format_baseline(rows: dict[tuple[str, str], int]) -> str:
    header = (
        "# McCabe (ruff C901) per-function ceilings. Values may only decrease.\n"
        f"# Format: <path>:<qualified_name> <max_complexity>\n"
        f"# Soft threshold for unlisted functions: {SOFT_THRESHOLD}\n"
        "# Regenerate after reductions: uv run python scripts/check_mccabe_ratchet.py --write\n"
    )
    lines = [
        f"{path}:{name} {complexity}"
        for (path, name), complexity in sorted(
            rows.items(), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    return header + "\n".join(lines) + ("\n" if lines else "")


def current_above_threshold(
    measured: dict[tuple[str, str], int] | None = None,
) -> dict[tuple[str, str], int]:
    measured = measured if measured is not None else _ruff_complexities(["src"])
    return {key: value for key, value in measured.items() if value >= SOFT_THRESHOLD}


def check(measured: dict[tuple[str, str], int] | None = None) -> int:
    """Require the baseline to match current complexities (>= soft threshold).

    - A listed function whose complexity rose above its recorded value fails.
    - A new function at or above the soft threshold fails until reviewed.
    - Stale high baseline rows (complexity fell, or the function disappeared)
      also fail so the checked-in file must be lowered via ``--write``.
    """
    measured = measured if measured is not None else _ruff_complexities(["src"])
    baseline = _entries(BASELINE)
    current = current_above_threshold(measured)

    regressions: list[str] = []
    for key in sorted(baseline.keys() | current.keys()):
        path, name = key
        allowed = baseline.get(key)
        got = current.get(key)
        if allowed is None and got is not None:
            regressions.append(
                f"  {path}:{name} complexity {got} is new (>= {SOFT_THRESHOLD}); "
                "reduce it or add via --write after review"
            )
        elif got is None and allowed is not None:
            measured_now = measured.get(key)
            detail = (
                f"now {measured_now}" if measured_now is not None else "missing (renamed/removed)"
            )
            regressions.append(
                f"  {path}:{name} baseline={allowed} but {detail}; lower/remove via --write"
            )
        elif got is not None and allowed is not None and got > allowed:
            regressions.append(f"  {path}:{name} complexity {got} > baseline {allowed}")
        elif got is not None and allowed is not None and got < allowed:
            regressions.append(
                f"  {path}:{name} complexity {got} < baseline {allowed}; "
                "lower the baseline via --write"
            )

    expected = _format_baseline(baseline)
    if BASELINE.read_text() != expected:
        regressions.append("  quality/mccabe_baseline.txt must be sorted unique canonical form")

    if regressions:
        print("mccabe complexity ratchet failed:", file=sys.stderr)
        for line in regressions:
            print(line, file=sys.stderr)
        return 1
    print(
        f"mccabe ratchet ok ({len(baseline)} baselined functions, soft threshold {SOFT_THRESHOLD})"
    )
    return 0


def write_baseline(measured: dict[tuple[str, str], int] | None = None) -> int:
    measured = measured if measured is not None else _ruff_complexities(["src"])
    current = current_above_threshold(measured)
    if BASELINE.exists():
        previous = _entries(BASELINE)
        raised = {
            key: (previous[key], current[key])
            for key in previous.keys() & current.keys()
            if current[key] > previous[key]
        }
        if raised:
            print("refusing to raise baseline values:", file=sys.stderr)
            for (path, name), (old, new) in sorted(raised.items()):
                print(f"  {path}:{name} {old} -> {new}", file=sys.stderr)
            return 1
        # New keys at/above the soft threshold are allowed on --write only as
        # an explicit acknowledgment (CI ``check`` still fails without it).
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(_format_baseline(current))
    print(f"wrote {BASELINE.relative_to(ROOT)} ({len(current)} functions)")
    return 0


def write_baseline_from_census(census_path: Path) -> int:
    """Pin pre-existing debt without ever raising an existing pin.

    ``--write`` is all-or-nothing: one regression (a function more complex than
    its recorded pin) blocks the write, so new-but-legitimate keys can never be
    pinned. This mode is the narrow exception. It is fail-closed on three axes:

    1. **Stale census.** The census digest is recomputed from its own payload
       and compared to the recorded ``digest``. A hand-edited or truncated
       census fails before any write.
    2. **Stale tree.** A fresh scan must reproduce the census exactly. If the
       tree moved since the census was taken, the pin is refused — the operator
       regenerates the census rather than blessing a tree they did not measure.
    3. **No raises.** Existing pins are carried over verbatim. A function that
       is currently more complex than its pin stays at the pin, so ``check``
       keeps failing on it. This mode can only ever *widen* the baseline to
       cover debt that was never pinned; it can never move a pin upward.
    """
    census = load_census(census_path)
    violations = census["violations"]
    ceiling = int(census["ceiling"])
    scope = list(census["scope"])  # type: ignore[arg-type]
    tool = str(census["tool"])

    recomputed = _census_digest(violations, ceiling, scope, tool)  # type: ignore[arg-type]
    if recomputed != census["digest"]:
        print(
            "refusing census: digest mismatch (census was hand-edited); "
            f"recorded {census['digest']}, recomputed {recomputed}",
            file=sys.stderr,
        )
        return 1

    findings = _ruff_findings(scope)
    fresh = build_census(findings, ceiling=ceiling, scope=scope)
    if fresh["digest"] != census["digest"]:
        print(
            "refusing census: tree has moved since the census was generated; "
            f"census {census['digest']} != fresh scan {fresh['digest']}. "
            "Regenerate with --write-census and re-review.",
            file=sys.stderr,
        )
        return 1

    measured = {(str(f["file"]), str(f["function"])): int(f["complexity"]) for f in findings}
    current = current_above_threshold(measured)
    previous = _entries(BASELINE) if BASELINE.exists() else {}
    pins = dict(current)
    # Existing pins win: carry the recorded (lower) value, never the fresh one.
    regressions: list[str] = []
    for key, allowed in previous.items():
        pins[key] = allowed
        got = current.get(key)
        if got is None:
            detail = f"now {measured[key]}" if key in measured else "missing (renamed/removed)"
            regressions.append(f"  {key[0]}:{key[1]} baseline={allowed} but {detail}")
        elif got > allowed:
            regressions.append(f"  {key[0]}:{key[1]} complexity {got} > baseline {allowed}")

    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(_format_baseline(pins))
    added = len(current.keys() - previous.keys())
    print(
        f"wrote {BASELINE.relative_to(ROOT)} from census {census_path.name}: "
        f"{added} pinned, {len(previous)} existing pins carried unchanged "
        f"({len(pins)} functions total)"
    )
    if regressions:
        print(
            "note: these pins are now BELOW current complexity and remain "
            "failing under check until fixed in src/ (this mode does not bless "
            "regressions):",
            file=sys.stderr,
        )
        for line in regressions:
            print(line, file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite quality/mccabe_baseline.txt from the current tree "
        "(refuses to raise any recorded value)",
    )
    parser.add_argument(
        "--write-census",
        type=Path,
        metavar="PATH",
        help="write a self-describing complexity census (JSON) to PATH; "
        "records every function above the ceiling plus a reproducible digest",
    )
    parser.add_argument(
        "--allow-with-census",
        type=Path,
        metavar="CENSUS_JSON",
        help="rewrite the baseline from a previously generated census, adding "
        "never-pinned functions at their measured value. Existing pins are "
        "carried unchanged and never raised. Fails closed if the census digest "
        "does not match its payload or a fresh scan of the tree",
    )
    args = parser.parse_args(argv)

    if args.write_census:
        census = build_census()
        args.write_census.parent.mkdir(parents=True, exist_ok=True)
        args.write_census.write_text(
            json.dumps(census, indent=2, sort_keys=True) + os.linesep, encoding="utf-8"
        )
        print(
            f"wrote {args.write_census} ({census['violation_count']} functions "
            f"above ceiling {census['ceiling']})"
        )
        return 0
    if args.allow_with_census:
        if args.write:
            print("--write and --allow-with-census are mutually exclusive", file=sys.stderr)
            return 2
        return write_baseline_from_census(args.allow_with_census)

    measured = _ruff_complexities(SCOPE)
    if args.write:
        return write_baseline(measured)
    return check(measured)


if __name__ == "__main__":
    raise SystemExit(main())
