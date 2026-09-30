from pathlib import Path
p = Path('scripts/verify_sota_finalization.py')
s = p.read_text()
old = '''    if not _under(resolved, run_root):
        raise VerificationError(f"{label} resolves outside manifest run directory: {value}")
    return resolved
'''
new = '''    if not lexical.exists():
        return lexical
    if not _under(resolved, run_root):
        raise VerificationError(f"{label} resolves outside manifest run directory: {value}")
    return resolved
'''
assert old in s
s = s.replace(old, new, 1)
old = '''def _verify_execution(group: dict[str, Any], manifest: dict[str, Any]) -> None:
    directory = Path(manifest["_manifest_dir"])
    command = _required(group, "command", list)
'''
new = '''def _verify_execution(group: dict[str, Any], manifest: dict[str, Any]) -> None:
    directory = Path(manifest["_manifest_dir"])
    command_record = _resolve_artifact(
        _required(group, "command_record", str), directory, manifest.get("repo"), label="command record"
    )
    if not command_record.is_file():
        raise VerificationError(f"missing command record for {group.get('group')}: {command_record}")
    recorded = _load_json(command_record)
    command = _required(group, "command", list)
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    if command_text != " ".join(command):
        raise VerificationError(f"command_text mismatch for {group.get('group')}")
    repo = _resolve_recorded(_required(manifest, "repo", str), directory)
'''
new = '''    if command_text != " ".join(command):
        raise VerificationError(f"command_text mismatch for {group.get('group')}")
    record_fields = (
        "group", "command", "command_text", "cwd", "shards", "stdout", "stderr", "transcript",
        "receipt", "receipt_sha256", "losses", "losses_sha256", "exit_code", "status",
    )
    for field in record_fields:
        if recorded.get(field) != group.get(field):
            raise VerificationError(f"command record mismatch for {group.get('group')}: {field}")
    repo = _resolve_recorded(_required(manifest, "repo", str), directory)
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    try:
        manifest = _load_json(path)
        manifest["_manifest_dir"] = str(path.parent)
'''
new = '''    try:
        if not path.is_file() or _has_reparse_any(path):
            raise VerificationError(f"manifest uses a symlink or reparse point or is missing: {path}")
        manifest = _load_json(path)
        manifest["_manifest_dir"] = str(path.parent)
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
