from pathlib import Path
p = Path('scripts/verify_sota_finalization.py')
s = p.read_text()
old = '''        if not _under(actual_lexical, input_root) or not _under(expected_lexical, input_root):
            raise VerificationError(f"shard path/relative mismatch or escape: {path_text}")
        if _has_reparse_component(actual_lexical, input_root) or _has_reparse_component(expected_lexical, input_root):
'''
new = '''        if not _under(actual_lexical, input_root) or not _under(expected_lexical, input_root):
            raise VerificationError(f"shard path/relative mismatch or escape: {path_text}")
        if not actual_lexical.exists():
            raise VerificationError(f"missing shard: {path_text}")
        if _has_reparse_component(actual_lexical, input_root) or _has_reparse_component(expected_lexical, input_root):
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
