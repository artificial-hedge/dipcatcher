"""Docs-vs-CLI drift guard: fx1 documentation may only reference commands,
targets, and import paths that actually exist. An auditor reading the docs
must be able to run what the docs say."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from typer.main import get_command

from fx1.cli import app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC_PATHS = sorted(
    [_REPO_ROOT / "README.md", *_REPO_ROOT.glob("docs/FX1*.md")],
)

# `fx1 ...` / `make ...` inside inline code or command-example fences.
_INLINE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_COMMAND_FENCE_LANGUAGES = {
    "",
    "bash",
    "bat",
    "cmd",
    "console",
    "fish",
    "plaintext",
    "powershell",
    "ps1",
    "sh",
    "shell",
    "shell-session",
    "text",
    "zsh",
}
_FX1_CMD = re.compile(r"(?<![/\w])fx1 ([a-z][a-z0-9-]*)(?: ([a-z][a-z0-9-]*))?\b")
_MAKE_TARGET = re.compile(r"\bmake ([a-z0-9][a-z0-9_-]*)")
_IMPORT_FROM = re.compile(r"from (fx1(?:\.[a-z_]+)+) import ([a-z_][a-zA-Z0-9_]*)")
_MODULE_REF = re.compile(r"`(fx1(?:\.[a-z_]+)+)`")


def _code_spans(text: str) -> list[str]:
    """Extract examples without treating diagram labels as shell commands.

    Consume every fence before looking for inline code: backticks in a Mermaid
    label or Python string are literal content, not Markdown code spans. Plain
    text and unlabeled fences remain covered for terminal-output examples.
    """
    spans: list[str] = []
    fence = ""
    command_block = False
    for line in text.splitlines():
        if fence:
            # Markdown permits a closing fence longer than its opening fence,
            # but not a shorter fence or one using the other marker character.
            if re.fullmatch(rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*", line):
                fence = ""
            elif command_block:
                spans.append(line)
            continue
        if match := _FENCE_OPEN.fullmatch(line):
            fence, info = match.groups()
            language = info.strip().split(maxsplit=1)[0] if info.strip() else ""
            command_block = language.lower() in _COMMAND_FENCE_LANGUAGES
        else:
            spans.extend(match.group(2) for match in _INLINE.finditer(line))
    return spans


def _cli_tree() -> dict[str, object]:
    root = get_command(app)
    tree: dict[str, object] = {}
    for name, cmd in root.commands.items():
        tree[name] = {sub for sub in cmd.commands} if hasattr(cmd, "commands") else set()
    return tree


def _makefile_targets() -> set[str]:
    text = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([a-z0-9][a-z0-9_-]*):(?:\s|##)", text, re.MULTILINE))


@pytest.mark.parametrize("doc", _DOC_PATHS, ids=lambda p: p.name)
def test_documented_fx1_commands_exist(doc: Path):
    tree = _cli_tree()
    text = doc.read_text(encoding="utf-8")
    unknown = []
    for span in _code_spans(text):
        for match in _FX1_CMD.finditer(span):
            first, second = match.group(1), match.group(2)
            if first not in tree:
                unknown.append(f"fx1 {first}")
            elif second and tree[first] and second not in tree[first]:
                unknown.append(f"fx1 {first} {second}")
    assert not unknown, f"{doc.name}: unknown fx1 commands {unknown}"


@pytest.mark.parametrize("doc", _DOC_PATHS, ids=lambda p: p.name)
def test_documented_make_targets_exist(doc: Path):
    targets = _makefile_targets()
    text = doc.read_text(encoding="utf-8")
    unknown = []
    for span in _code_spans(text):
        for match in _MAKE_TARGET.finditer(span):
            if match.group(1) not in targets:
                unknown.append(f"make {match.group(1)}")
    assert not unknown, f"{doc.name}: unknown make targets {unknown}"


@pytest.mark.parametrize("language", ["mermaid", "python", "json"])
def test_non_command_fences_do_not_leak_inline_code(language: str):
    text = (
        "Run `fx1 doctor`.\n"
        f"```{language}\n"
        'label = "`fx1 positive` and `make targets`"\n'
        "```\n"
        "Then run `make lint`.\n"
    )
    assert _code_spans(text) == ["fx1 doctor", "make lint"]


def test_documented_prose_is_not_a_command(tmp_path: Path):
    doc = tmp_path / "prose.md"
    doc.write_text(
        "fx1 positive examples, fx1 negative examples, fx1 modules, and make targets.\n"
        "`fx1` positive examples, `fx1` negative examples, `fx1` modules, `make` targets.\n"
        "```mermaid\n"
        '  Rel(researcher, harness, "make targets, dipcatcher CLI")\n'
        "  verified --> corpus: eligible receipt -> fx1 positive example\n"
        "  blocked --> corpus: ineligible receipt -> fx1 negative example\n"
        '  x-axis ["quant_fund modules", "fx1 modules", "test files"]\n'
        "```\n",
        encoding="utf-8",
    )
    test_documented_fx1_commands_exist(doc)
    test_documented_make_targets_exist(doc)


@pytest.mark.parametrize(
    "template",
    [
        "Run `{command}` next.",
        "Run ``{command}`` next.",
        "```bash\n{command}\n```",
        "```sh\n{command}\n```",
        "```console\n$ {command}\n```",
        "```text\n{command}\n```",
        "```\n{command}\n```",
        "~~~shell\n{command}\n~~~",
        "  ````bash\n{command}\n  `````",
    ],
)
@pytest.mark.parametrize(
    "command", ["fx1 not-a-command", "fx1 corpus not-a-command", "make not-a-target"]
)
def test_unknown_documented_commands_are_still_rejected(
    tmp_path: Path, template: str, command: str
):
    doc = tmp_path / "commands.md"
    doc.write_text(template.format(command=command), encoding="utf-8")
    check = (
        test_documented_make_targets_exist
        if command.startswith("make ")
        else test_documented_fx1_commands_exist
    )
    with pytest.raises(AssertionError, match=re.escape(command)):
        check(doc)


def test_fence_closers_must_match_marker_and_length():
    text = "````mermaid\n```\n`fx1 positive`\n~~~~\n`make targets`\n`````\nRun `fx1 doctor`.\n"
    assert _code_spans(text) == ["fx1 doctor"]


@pytest.mark.parametrize("doc", _DOC_PATHS, ids=lambda p: p.name)
def test_documented_import_paths_resolve(doc: Path):
    text = doc.read_text(encoding="utf-8")
    problems = []
    for module_name, symbol in _IMPORT_FROM.findall(text):
        module = importlib.import_module(module_name)
        if not hasattr(module, symbol):
            problems.append(f"{module_name}.{symbol}")
    for module_name in _MODULE_REF.findall(text):
        try:
            importlib.import_module(module_name)
        except ImportError:
            # Not a module — maybe a symbol on its parent (fx1.mrm.compile_dossier).
            parent, _, symbol = module_name.rpartition(".")
            try:
                module = importlib.import_module(parent)
                if not symbol or not hasattr(module, symbol):
                    problems.append(module_name)
            except ImportError:
                problems.append(module_name)
    assert not problems, f"{doc.name}: unresolvable imports {problems}"
