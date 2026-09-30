from pathlib import Path
p = Path('scripts/verify_sota_finalization.py')
s = p.read_text()
start = s.index('def _resolve_recorded(')
end = s.index('\n\ndef _under(', start)
new = '''def _recorded_candidates(value: str | os.PathLike[str], manifest_dir: Path, repo: str | None = None) -> list[Path]:
    """Return lexical candidates without dereferencing links."""
    raw = os.fspath(value)
    candidates: list[Path] = []
    native = Path(raw)
    if native.is_absolute():
        candidates.append(native)
    else:
        candidates.extend((manifest_dir / native, Path.cwd() / native))
    if _windows_absolute(raw):
        slash = raw.replace("\\\\", "/")
        drive, tail = slash[0].lower(), slash[2:].lstrip("/")
        candidates.append(Path("/") / drive / tail)
        if repo and _windows_absolute(repo):
            repo_slash = repo.replace("\\\\", "/").rstrip("/")
            if slash.lower().startswith(repo_slash.lower() + "/"):
                candidates.append(Path.cwd() / slash[len(repo_slash):].lstrip("/"))
    return [candidate.expanduser().absolute() for candidate in candidates]


def _resolve_recorded(value: str | os.PathLike[str], manifest_dir: Path, repo: str | None = None) -> Path:
    """Resolve native paths and Windows paths emitted by PowerShell."""
    candidates = _recorded_candidates(value, manifest_dir, repo)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(strict=False)
    return candidates[0].resolve(strict=False)


def _resolve_recorded_lexical(value: str | os.PathLike[str], manifest_dir: Path, repo: str | None = None) -> Path:
    """Resolve a recorded path while preserving link/reparse components."""
    candidates = _recorded_candidates(value, manifest_dir, repo)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
'''
p.write_text(s[:start] + new + s[end:])
