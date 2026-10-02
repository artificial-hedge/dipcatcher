"""SBOM generation tests."""

from pathlib import Path

import pytest

from fx1.sbom import generate_sbom


def test_sbom_from_real_uv_lock():
    lock = Path(__file__).parents[2] / "uv.lock"
    if not lock.exists():  # standalone overlay: synthesize a minimal lock
        lock = Path(__file__).parents[2] / "tests" / "fx1" / "_test_uv.lock"
        lock.write_text(
            '[[package]]\nname = "pydantic"\nversion = "2.11.4"\n'
            + "\n".join(f'[[package]]\nname = "pkg{i}"\nversion = "1.0.{i}"' for i in range(60)),
            encoding="utf-8",
        )
    sbom = generate_sbom(lock)
    assert len(sbom.entries) > 50
    assert len(sbom.lockfile_sha256) == 64
    names = [e.name for e in sbom.entries]
    assert "pydantic" in names
    assert names == sorted(names)


def test_sbom_fail_closed_missing_lock(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        generate_sbom(tmp_path / "nope.lock")
    bad = tmp_path / "bad.lock"
    bad.write_text("not a lockfile", encoding="utf-8")
    with pytest.raises(ValueError, match="no packages"):
        generate_sbom(bad)


def test_dynamic_editable_root_uses_canonical_version(tmp_path: Path):
    from fx1 import __version__

    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "fx-1"\nsource = { editable = "." }\n')
    entry = generate_sbom(lock).entries[0]
    assert (entry.name, entry.version, entry.sha256) == ("fx-1", __version__, None)


@pytest.mark.parametrize(
    "package",
    [
        'name = "dependency"',
        'name = "fx-1"',
        'name = "fx-1"\nsource = { registry = "https://example.invalid" }',
        'name = "fx-1"\nsource = { editable = "../other" }',
        'name = "other"\nsource = { editable = "." }',
        'version = "1.0"',
        'name = "dependency"\nversion = 1',
        'name = "dependency"\nversion = ""',
        'name = ""\nversion = "1.0"',
    ],
)
def test_unversioned_or_invalid_dependencies_fail_closed(tmp_path: Path, package: str):
    lock = tmp_path / "uv.lock"
    lock.write_text(f"[[package]]\n{package}\n")
    with pytest.raises(ValueError, match="malformed"):
        generate_sbom(lock)


def test_toml_syntax_and_exact_file_hash(tmp_path: Path):
    import hashlib

    lock = tmp_path / "uv.lock"
    raw = (
        "version = 1\r\n[[package]] # first package\r\n"
        "name='z'\r\nversion='2.0'\r\n"
        "sdist = { hash = 'sha256:" + "a" * 64 + "' }\r\n"
        "[[package]]\r\nname='a'\r\nversion='1.0'\r\n"
        "wheels = [{ hash = 'sha256:" + "b" * 64 + "' }]\r\n"
    ).encode()
    lock.write_bytes(raw)
    sbom = generate_sbom(lock)
    assert [(e.name, e.version, e.sha256) for e in sbom.entries] == [
        ("a", "1.0", "b" * 64),
        ("z", "2.0", "a" * 64),
    ]
    assert sbom.lockfile_sha256 == hashlib.sha256(raw).hexdigest()


def test_nested_version_cannot_mask_missing_package_version(tmp_path: Path):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname="dependency"\n[package.metadata]\nversion="1.0"\n')
    with pytest.raises(ValueError, match="malformed"):
        generate_sbom(lock)


def test_explicit_root_version_is_preserved(tmp_path: Path):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname="fx-1"\nversion="0.1.0"\nsource={editable="."}\n')
    assert generate_sbom(lock).entries[0].version == "0.1.0"


@pytest.mark.parametrize("text", ["version=1\n", '[package]\nname="x"\nversion="1"\n'])
def test_missing_package_array_fails_closed(tmp_path: Path, text: str):
    lock = tmp_path / "uv.lock"
    lock.write_text(text)
    with pytest.raises(ValueError, match="no packages"):
        generate_sbom(lock)


def test_empty_package_cannot_be_silently_omitted(tmp_path: Path):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname="valid"\nversion="1"\n[[package]]\n')
    with pytest.raises(ValueError, match="malformed"):
        generate_sbom(lock)


def test_nested_metadata_hash_is_not_an_artifact_hash(tmp_path: Path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname="dependency"\nversion="1"\n'
        '[package.metadata]\nhash="sha256:' + "a" * 64 + '"\n'
    )
    assert generate_sbom(lock).entries[0].sha256 is None


def test_sbom_audit_handles_rejected_invalid_toml():
    from fx1.sbom_audit import sbom_audit

    report = sbom_audit()
    assert report["n_entries"] == 2
    for key in ("sorted", "hash_carried", "hashless_none", "lockfile_sha", "injection_contained"):
        assert report[key] is True
    assert report["missing_raises"] == "raise:FileNotFoundError"
    for key in ("no_packages_raises", "half_block_raises", "injection_raises"):
        assert report[key] == "raise:ValueError"
