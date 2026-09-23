"""Lossless byte corpus of a git working tree.

Every regular file discovered from git is written as a length-prefixed frame.
The frame format is pure bytes (vocabulary 256):

    b"DCF1" + path_len:u32be + path_utf8 + content_len:u64be + content

Parsing walks those lengths from the start of the stream, so a file body may
contain the magic bytes without ending the frame. The only paths left out are
the converter's own output directory (the weights are the product, not an
input) and ``artifacts/repo_llm/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MAGIC = b"DCF1"
VOCAB_SIZE = 256
CORPUS_FILENAME = "corpus.bin"
MANIFEST_FILENAME = "manifest.json"
OUTPUT_PREFIX = "artifacts/repo_llm/"
_CHUNK = 1 << 20


class RepoLLMError(ValueError):
    """The repository corpus or checkpoint is not usable."""


@dataclass(frozen=True)
class FileRecord:
    path: str
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class CorpusManifest:
    files: tuple[FileRecord, ...]
    token_count: int
    corpus_sha256: str
    excluded_prefixes: tuple[str, ...]
    git_head: str | None
    vocab_size: int = VOCAB_SIZE

    def to_json(self) -> dict[str, object]:
        return {
            "version": 1,
            "vocab_size": self.vocab_size,
            "endian": "byte-stream",
            "frame": "DCF1 + u32be path_len + path + u64be content_len + content",
            "token_count": self.token_count,
            "corpus_sha256": self.corpus_sha256,
            "excluded_prefixes": list(self.excluded_prefixes),
            "git_head": self.git_head,
            "file_count": len(self.files),
            "files": [
                {"path": rec.path, "bytes": rec.byte_length, "sha256": rec.sha256}
                for rec in self.files
            ],
        }


def discover_repo_files(root: Path, out_dir: Path | None = None) -> list[str]:
    """Return sorted repo-relative paths for every file the corpus must include."""
    root = root.resolve()
    prefixes = _exclude_prefixes(root, out_dir)
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise RepoLLMError(f"git ls-files failed in {root}: {err}")
    found: list[str] = []
    for entry in proc.stdout.split(b"\0"):
        if not entry:
            continue
        rel = entry.decode("utf-8")
        posix = rel.replace("\\", "/")
        if _excluded(posix, prefixes):
            continue
        path = _safe_file(root, posix)
        if not path.is_file():
            raise RepoLLMError(f"git path is not a regular file: {posix}")
        found.append(posix)
    found.sort()
    return found


def build_corpus(
    root: Path,
    out_dir: Path,
    files: list[str] | None = None,
) -> CorpusManifest:
    """Write ``corpus.bin`` and ``manifest.json`` for ``files`` or the whole repo."""
    root = root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefixes = _exclude_prefixes(root, out_dir)
    selected = files if files is not None else discover_repo_files(root, out_dir)
    ordered = sorted(set(selected))
    if not ordered:
        raise RepoLLMError(f"no files to convert under {root}")
    for rel in ordered:
        if _excluded(rel, prefixes):
            raise RepoLLMError(f"refusing to train on converter output: {rel}")
        _safe_file(root, rel)

    partial = out_dir / f"{CORPUS_FILENAME}.partial"
    corpus_path = out_dir / CORPUS_FILENAME
    writer = hashlib.sha256()
    records: list[FileRecord] = []
    token_count = 0
    print(f"framing {len(ordered)} files into {corpus_path}", flush=True)
    with partial.open("wb") as handle:
        for index, rel in enumerate(ordered, start=1):
            path = root / rel
            if not path.is_file():
                raise RepoLLMError(f"not a regular file: {rel}")
            path_b = rel.encode("utf-8")
            file_hash = hashlib.sha256()
            byte_length = 0
            header = (
                MAGIC
                + len(path_b).to_bytes(4, "big")
                + path_b
                + path.stat().st_size.to_bytes(8, "big")
            )
            _write(handle, writer, header)
            token_count += len(header)
            with path.open("rb") as src:
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    file_hash.update(chunk)
                    _write(handle, writer, chunk)
                    byte_length += len(chunk)
                    token_count += len(chunk)
            if byte_length != path.stat().st_size:
                raise RepoLLMError(f"file changed while reading: {rel}")
            records.append(
                FileRecord(path=rel, byte_length=byte_length, sha256=file_hash.hexdigest())
            )
            if byte_length >= 50_000_000 or index == len(ordered) or index % 500 == 0:
                print(f"framed {index}/{len(ordered)} {rel} ({byte_length} bytes)", flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, corpus_path)
    digest = _sha256_file(corpus_path)
    if digest != writer.hexdigest():
        raise RepoLLMError("corpus hash on disk does not match the bytes written")
    if token_count != corpus_path.stat().st_size:
        raise RepoLLMError("corpus size does not match the framed token count")
    manifest = CorpusManifest(
        files=tuple(records),
        token_count=token_count,
        corpus_sha256=digest,
        excluded_prefixes=prefixes,
        git_head=_git_head(root),
    )
    verify_corpus(root, corpus_path, manifest)
    _write_manifest(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


def ensure_corpus(
    root: Path,
    out_dir: Path,
    files: list[str] | None = None,
) -> CorpusManifest:
    """Reuse a corpus when every source file still matches the manifest."""
    root = root.resolve()
    manifest_path = out_dir / MANIFEST_FILENAME
    corpus_path = out_dir / CORPUS_FILENAME
    selected = files if files is not None else discover_repo_files(root, out_dir)
    ordered = sorted(set(selected))
    if manifest_path.is_file() and corpus_path.is_file():
        existing = load_manifest(manifest_path)
        same_list = [rec.path for rec in existing.files] == ordered
        same_bytes = same_list and all(_record_matches(root, rec) for rec in existing.files)
        same_corpus = same_bytes and _sha256_file(corpus_path) == existing.corpus_sha256
        if same_corpus and corpus_path.stat().st_size == existing.token_count:
            return existing
    return build_corpus(root, out_dir, ordered)


def load_manifest(path: Path) -> CorpusManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise RepoLLMError(f"unsupported corpus manifest version in {path}")
    files = tuple(
        FileRecord(path=str(row["path"]), byte_length=int(row["bytes"]), sha256=str(row["sha256"]))
        for row in payload["files"]
    )
    return CorpusManifest(
        files=files,
        token_count=int(payload["token_count"]),
        corpus_sha256=str(payload["corpus_sha256"]),
        excluded_prefixes=tuple(str(p) for p in payload["excluded_prefixes"]),
        git_head=payload.get("git_head"),
        vocab_size=int(payload.get("vocab_size", VOCAB_SIZE)),
    )


def open_corpus(path: Path, token_count: int) -> NDArray[np.uint8]:
    if token_count < 1:
        raise RepoLLMError("corpus is empty")
    if path.stat().st_size != token_count:
        raise RepoLLMError(f"corpus file size does not match token_count for {path}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(token_count,))


def verify_corpus(root: Path, corpus_path: Path, manifest: CorpusManifest) -> None:
    """Check that walking the frames reproduces every source file."""
    root = root.resolve()
    if corpus_path.stat().st_size != manifest.token_count:
        raise RepoLLMError("corpus length does not match the manifest")
    blob = open_corpus(corpus_path, manifest.token_count)
    offset = 0
    seen = 0
    for rec in manifest.files:
        if offset + len(MAGIC) > manifest.token_count:
            raise RepoLLMError(f"truncated frame before {rec.path}")
        if _bytes_at(blob, offset, len(MAGIC)) != MAGIC:
            raise RepoLLMError(f"missing DCF1 magic before {rec.path}")
        offset += len(MAGIC)
        path_len = int.from_bytes(_bytes_at(blob, offset, 4), "big")
        offset += 4
        path = _bytes_at(blob, offset, path_len).decode("utf-8")
        offset += path_len
        content_len = int.from_bytes(_bytes_at(blob, offset, 8), "big")
        offset += 8
        if path != rec.path or content_len != rec.byte_length:
            raise RepoLLMError(f"frame header does not match manifest for {rec.path}")
        if not _span_matches(blob, offset, root / rec.path, content_len):
            raise RepoLLMError(f"frame body does not match {rec.path}")
        offset += content_len
        seen += 1
    if seen != len(manifest.files) or offset != manifest.token_count:
        raise RepoLLMError("corpus frames did not consume the token stream")


def _span_matches(blob: NDArray[np.uint8], offset: int, path: Path, length: int) -> bool:
    with path.open("rb") as handle:
        pos = 0
        while pos < length:
            chunk = handle.read(min(_CHUNK, length - pos))
            if not chunk:
                return False
            got = _bytes_at(blob, offset + pos, len(chunk))
            if got != chunk:
                return False
            pos += len(chunk)
    return True


def _bytes_at(blob: NDArray[np.uint8], offset: int, length: int) -> bytes:
    if length < 0 or offset < 0 or offset + length > int(blob.shape[0]):
        raise RepoLLMError("frame points outside the corpus")
    return np.ascontiguousarray(blob[offset : offset + length]).tobytes()


def _record_matches(root: Path, rec: FileRecord) -> bool:
    path = root / rec.path
    if not path.is_file() or path.stat().st_size != rec.byte_length:
        return False
    return _sha256_file(path) == rec.sha256


def _write(handle, digest, payload: bytes) -> None:
    handle.write(payload)
    digest.update(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: CorpusManifest) -> None:
    text = json.dumps(manifest.to_json(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _exclude_prefixes(root: Path, out_dir: Path | None) -> tuple[str, ...]:
    prefixes = [OUTPUT_PREFIX]
    if out_dir is None:
        return tuple(prefixes)
    try:
        rel = out_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return tuple(prefixes)
    if rel.parts and rel.parts != (".",):
        rel_s = rel.as_posix().rstrip("/") + "/"
        if rel_s not in prefixes:
            prefixes.append(rel_s)
    return tuple(prefixes)


def _excluded(posix: str, prefixes: tuple[str, ...]) -> bool:
    return any(posix == prefix.rstrip("/") or posix.startswith(prefix) for prefix in prefixes)


def _safe_file(root: Path, rel: str) -> Path:
    if not rel or rel.startswith("/") or "\\" in rel or rel.split("/")[0] == "..":
        raise RepoLLMError(f"refusing unsafe relative path: {rel}")
    path = (root / rel).resolve()
    root_r = root.resolve()
    if path != root_r and root_r not in path.parents:
        raise RepoLLMError(f"path escapes repository: {rel}")
    return path


def _git_head(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None
