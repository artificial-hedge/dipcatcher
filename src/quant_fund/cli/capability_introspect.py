"""Click and Typer introspection for lab command arguments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from quant_fund.cli.capability_model import LabCommand, _nonportable, _number_or_none, _order_arg


def _collect(root: Any) -> list[LabCommand]:
    """Walk groups. A one-command Typer app is the command itself, not a group."""
    found = _walk(root, ())
    if found or isinstance(getattr(root, "commands", None), dict):
        return found
    name = getattr(root, "name", None)
    if not name:
        return []
    return [_leaf(root, (str(name),))]


def _walk(node: Any, prefix: tuple[str, ...]) -> list[LabCommand]:
    found: list[LabCommand] = []
    children = getattr(node, "commands", None)
    if not isinstance(children, dict):
        return found
    for name in children:
        child = children[name]
        argv = prefix + (str(name),)
        nested = getattr(child, "commands", None)
        if isinstance(nested, dict) and nested:
            found.extend(_walk(child, argv))
            continue
        found.append(_leaf(child, argv))
    return found


def _leaf(node: Any, argv: tuple[str, ...]) -> LabCommand:
    params = getattr(node, "params", ()) or ()
    args = tuple(_arg_from_param(param) for param in params if not _skip_param(param))
    return LabCommand(
        argv=argv,
        help_text=_clip_help(getattr(node, "help", None)),
        hidden=bool(getattr(node, "hidden", False)),
        args=args,
    )


def _skip_param(param: Any) -> bool:
    """Ignore Click's help option and Typer's shell-completion installers."""
    name = str(getattr(param, "name", ""))
    return name in {"help", "install_completion", "show_completion"}


def _arg_from_param(param: Any) -> dict[str, Any]:
    kind = _kind(param)
    name = _public_name(param, kind)
    schema_type = _schema_type(param)
    arg: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "type": schema_type,
        "required": bool(getattr(param, "required", False)),
        "default": _default_text(getattr(param, "default", None)),
        "repeatable": bool(getattr(param, "multiple", False)),
        "help": _clip_help(getattr(param, "help", None)),
    }
    choices = _choice_list(param)
    if choices is not None:
        arg["choices"] = choices
    lower, upper = _range_bounds(param)
    if lower is not None:
        arg["min"] = lower
    if upper is not None:
        arg["max"] = upper
    if schema_type == "path":
        arg["path_role"] = _path_role(name)
    return _order_arg(arg)


def _kind(param: Any) -> str:
    if "argument" in type(param).__name__.lower():
        return "argument"
    opts = [str(opt) for opt in (getattr(param, "opts", ()) or ())]
    if opts and not opts[0].startswith("-"):
        return "argument"
    return "option"


def _public_name(param: Any, kind: str) -> str:
    opts = [str(opt) for opt in (getattr(param, "opts", ()) or ())]
    if kind == "option":
        longs = [opt for opt in opts if opt.startswith("--")]
        if longs:
            return max(longs, key=len)
        shorts = [opt for opt in opts if opt.startswith("-")]
        if shorts:
            return max(shorts, key=len)
    if opts:
        return opts[0]
    return str(getattr(param, "name", "arg"))


def _schema_type(param: Any) -> str:
    if bool(getattr(param, "is_flag", False)):
        return "flag"
    type_name = type(getattr(param, "type", object())).__name__.lower()
    if "choice" in type_name:
        return "enum"
    if "path" in type_name:
        return "path"
    if "float" in type_name:
        return "float"
    if "int" in type_name:
        return "int"
    return "str"


def _choice_list(param: Any) -> list[str] | None:
    annotation = getattr(param, "type", None)
    if annotation is None or "choice" not in type(annotation).__name__.lower():
        return None
    raw = getattr(annotation, "choices", None)
    if not raw:
        return None
    return [str(item) for item in raw]


def _range_bounds(param: Any) -> tuple[int | float | None, int | float | None]:
    annotation = getattr(param, "type", None)
    if annotation is None or "range" not in type(annotation).__name__.lower():
        return None, None
    return (
        _number_or_none(getattr(annotation, "min", None)),
        _number_or_none(getattr(annotation, "max", None)),
    )


def _path_role(name: str) -> str:
    token = name.lstrip("-").lower().replace("_", "-")
    parts = token.split("-")
    if "config" in parts:
        return "config"
    if token in {"out", "output", "out-dir", "output-dir"} or token.endswith("-out"):
        return "output_dir"
    if "out-dir" in token or token.endswith("-output"):
        return "output_dir"
    return "input"


def _default_text(value: object) -> str | None:
    if value is None or type(value).__name__ in {"Sentinel", "UNSET"}:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    text = _path_text(value) if isinstance(value, os.PathLike) else _plain_text(value)
    if text is None or _nonportable(text):
        return None
    return text


def _plain_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [
            _plain_text(item) if not isinstance(item, os.PathLike) else _path_text(item)
            for item in value
        ]
        if any(part is None for part in parts):
            return None
        return ",".join(str(part) for part in parts)
    return None


def _path_text(value: os.PathLike[str] | os.PathLike[bytes]) -> str:
    raw = os.fspath(value)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return Path(raw).as_posix()


def _clip_help(text: object) -> str:
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if not stripped:
        return ""
    collapsed = " ".join(stripped.splitlines()[0].split())
    if len(collapsed) <= 160:
        return collapsed
    return collapsed[:157] + "..."
