from pathlib import Path
p = Path('scripts/verify_sota_finalization.py')
s = p.read_text()
needle = '''def _has_reparse_component(path: Path, root: Path) -> bool:
'''
insert = '''def _has_reparse_any(path: Path) -> bool:
    """Return whether any existing component of a path is a link/reparse point."""
    absolute = path.absolute()
    current = Path(absolute.anchor) if absolute.anchor else Path('.')
    for part in absolute.parts:
        if part == absolute.anchor:
            continue
        current /= part
        try:
            if current.is_symlink():
                return True
            if os.name == "nt":
                attributes = os.stat(current, follow_symlinks=False).st_file_attributes
                if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                    return True
        except OSError:
            return True
    return False


'''
assert needle in s
s = s.replace(needle, insert + needle, 1)
old = '''def verify_manifest(manifest_path: str | os.PathLike[str], *, proof_grade: bool = False) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    errors: list[str] = []
'''
new = '''def verify_manifest(manifest_path: str | os.PathLike[str], *, proof_grade: bool = False) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().absolute()
    errors: list[str] = []
'''
assert old in s
s = s.replace(old, new, 1)
old = '''        run_id = _required(manifest, "run_id", str)
        if not _SAFE_RUN_ID.fullmatch(run_id) or path.parent.name != run_id:
            raise VerificationError("unsafe or directory-mismatched run_id")
        input_root = _resolve_recorded(_required(manifest, "input_dir", str), path.parent, manifest.get("repo"))
        if not input_root.is_dir():
            raise VerificationError(f"missing input directory: {input_root}")
        groups = _required(manifest, "groups", list)
'''
new = '''        run_id = _required(manifest, "run_id", str)
        if not _SAFE_RUN_ID.fullmatch(run_id) or path.parent.name != run_id:
            raise VerificationError("unsafe or directory-mismatched run_id")
        repo_text = _required(manifest, "repo", str)
        repo_lexical = _resolve_recorded_lexical(repo_text, path.parent)
        if not repo_lexical.is_dir():
            raise VerificationError(f"missing repository directory: {repo_lexical}")
        if _has_reparse_any(repo_lexical):
            raise VerificationError(f"repository uses a symlink or reparse point: {repo_lexical}")
        input_text = _required(manifest, "input_dir", str)
        input_lexical = _resolve_recorded_lexical(input_text, path.parent, repo_text)
        if not input_lexical.is_dir():
            raise VerificationError(f"missing input directory: {input_lexical}")
        if not _under(input_lexical, repo_lexical):
            raise VerificationError(f"input directory escapes repository: {input_text}")
        if _has_reparse_any(input_lexical):
            raise VerificationError(f"input directory uses a symlink or reparse point: {input_lexical}")
        expected_run = input_lexical / "runs" / run_id
        if path.parent != expected_run:
            raise VerificationError("manifest is not under repository input_dir/runs/run_id")
        if _has_reparse_any(path.parent):
            raise VerificationError(f"manifest run directory uses a symlink or reparse point: {path.parent}")
        input_root = input_lexical.resolve(strict=False)
        groups = _required(manifest, "groups", list)
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
