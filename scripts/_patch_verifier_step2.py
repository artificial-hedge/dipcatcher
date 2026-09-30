from pathlib import Path
p = Path('scripts/verify_sota_finalization.py')
s = p.read_text()
old = '''    resolved = _resolve_recorded(value, manifest_dir, repo)
    run_root = manifest_dir.resolve(strict=False)
    if not _under(resolved, run_root):
        raise VerificationError(f"{label} escapes manifest run directory: {value}")
    if _has_reparse_component(resolved, run_root):
        raise VerificationError(f"{label} uses a symlink or reparse point: {value}")
    return resolved
'''
new = '''    lexical = _resolve_recorded_lexical(value, manifest_dir, repo)
    run_root = manifest_dir.absolute()
    if not _under(lexical, run_root):
        raise VerificationError(f"{label} escapes manifest run directory: {value}")
    if _has_reparse_component(lexical, run_root):
        raise VerificationError(f"{label} uses a symlink or reparse point: {value}")
    resolved = lexical.resolve(strict=False)
    if not _under(resolved, run_root):
        raise VerificationError(f"{label} resolves outside manifest run directory: {value}")
    return resolved
'''
assert old in s
s = s.replace(old, new, 1)
old = '''        actual = _resolve_recorded(path_text, manifest_dir, manifest.get("repo"))
        repo_text = manifest.get("repo")
        if not isinstance(repo_text, str):
            raise VerificationError("manifest repo is required for shard relative paths")
        repo_root = _resolve_recorded(repo_text, manifest_dir)
        expected = (repo_root / relative).resolve(strict=False)
        if actual != expected or not _under(actual, input_root) or not _under(expected, input_root):
            raise VerificationError(f"shard path/relative mismatch or escape: {path_text}")
        if _has_reparse_component(actual, input_root):
            raise VerificationError(f"shard uses a symlink or reparse point: {path_text}")
'''
new = '''        actual_lexical = _resolve_recorded_lexical(path_text, manifest_dir, manifest.get("repo"))
        repo_text = manifest.get("repo")
        if not isinstance(repo_text, str):
            raise VerificationError("manifest repo is required for shard relative paths")
        repo_root = _resolve_recorded(repo_text, manifest_dir)
        expected_lexical = repo_root / relative
        if not _under(actual_lexical, input_root) or not _under(expected_lexical, input_root):
            raise VerificationError(f"shard path/relative mismatch or escape: {path_text}")
        if _has_reparse_component(actual_lexical, input_root) or _has_reparse_component(expected_lexical, input_root):
            raise VerificationError(f"shard uses a symlink or reparse point: {path_text}")
        actual = actual_lexical.resolve(strict=False)
        expected = expected_lexical.resolve(strict=False)
        if actual != expected or not _under(actual, input_root) or not _under(expected, input_root):
            raise VerificationError(f"shard path/relative mismatch or escape: {path_text}")
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    transcript = _resolve_recorded(group["transcript"], directory, manifest.get("repo")).read_text(encoding="utf-8-sig")
    if not transcript.startswith("COMMAND: ") or "EXIT_CODE: 0" not in transcript:
        raise VerificationError(f"transcript does not bind command and successful exit for {group.get('group')}")
'''
new = '''    transcript_path = _resolve_artifact(group["transcript"], directory, manifest.get("repo"), label="transcript")
    transcript = transcript_path.read_text(encoding="utf-8-sig")
    lines = transcript.splitlines()
    if len(lines) < 2 or lines[0] != f"COMMAND: {command_text}":
        raise VerificationError(f"transcript command does not bind recorded argv for {group.get('group')}")
    match = re.fullmatch(r"EXIT_CODE: (-?\\d+)", lines[1])
    if match is None or int(match.group(1)) != 0 or group.get("exit_code") != 0:
        raise VerificationError(f"transcript does not bind command and successful exit for {group.get('group')}")
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    if sorted(str(v).lower() for v in source.values()) != sorted(s["sha256"] for s in shards):
        raise VerificationError("receipt source-part hashes do not match manifest shards")
'''
new = '''    expected_source = {s["relative"]: s["sha256"] for s in shards}
    normalized_source = {str(key).replace("\\\\", "/"): str(value).lower() for key, value in source.items()}
    if normalized_source != expected_source:
        raise VerificationError("receipt source-part paths or hashes do not match manifest shards")
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
