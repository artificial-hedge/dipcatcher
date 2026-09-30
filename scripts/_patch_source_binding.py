from pathlib import Path
p = Path('scripts/verify_sota_finalization.py')
s = p.read_text()
needle = '''def _under(path: Path, root: Path) -> bool:
'''
insert = '''def _provenance_key(value: str) -> str:
    """Normalize a recorded path key without resolving filesystem links."""
    key = value.replace("\\\\", "/")
    if _windows_absolute(key):
        return key.lower()
    return key


'''
assert needle in s
s = s.replace(needle, insert + needle, 1)
old = '''    expected_source = {s["relative"]: s["sha256"] for s in shards}
    normalized_source = {str(key).replace("\\\\", "/"): str(value).lower() for key, value in source.items()}
    if normalized_source != expected_source:
'''
new = '''    expected_source = {_provenance_key(str(s["path"])): s["sha256"] for s in shards}
    normalized_source = {_provenance_key(str(key)): str(value).lower() for key, value in source.items()}
    if normalized_source != expected_source:
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)

p = Path('tests/unit/test_sota_finalization_verifier.py')
s = p.read_text()
s = s.replace('"source_parts_sha256": {"eval/part.losses.npz": sha(shard)},', '"source_parts_sha256": {str(shard): sha(shard)},')
p.write_text(s)
