"""Corpus coverage and trainable repository language-model weights."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from quant_fund.repo_llm import (
    OUTPUT_PREFIX,
    build_corpus,
    discover_repo_files,
    fold_lanes,
)
from quant_fund.repo_llm.corpus import verify_corpus

ROOT = Path(__file__).resolve().parents[2]


def test_fold_every_byte_changes_its_lane() -> None:
    base = np.arange(50, dtype=np.uint8)
    lanes = fold_lanes(base, n_lanes=4, seed=1)
    for index in range(base.size):
        alt = base.copy()
        alt[index] = np.uint8(int(alt[index]) ^ 1)
        assert not np.array_equal(fold_lanes(alt, n_lanes=4, seed=1), lanes)


def test_frame_roundtrip_includes_magic_inside_body(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    payload = b"DCF1" + bytes(range(256)) + b"\x00\xff"
    (root / "bin.dat").write_bytes(payload)
    (root / "empty").write_bytes(b"")
    nested = root / "notes"
    nested.mkdir()
    (nested / "readme.txt").write_text("alpha\n", encoding="utf-8")
    out = tmp_path / "out"
    manifest = build_corpus(
        root,
        out,
        files=["notes/readme.txt", "bin.dat", "empty"],
    )
    assert [rec.path for rec in manifest.files] == ["bin.dat", "empty", "notes/readme.txt"]
    assert manifest.files[0].byte_length == len(payload)
    verify_corpus(root, out / "corpus.bin", manifest)
    again = build_corpus(root, out, files=["bin.dat", "empty", "notes/readme.txt"])
    assert again.corpus_sha256 == manifest.corpus_sha256


def test_discover_covers_tracked_files_and_skips_output(tmp_path: Path) -> None:
    junk_dir = ROOT / "artifacts" / "repo_llm"
    junk_dir.mkdir(parents=True, exist_ok=True)
    junk = junk_dir / "ignore_me.bin"
    junk.write_bytes(b"not part of the corpus")
    try:
        found = discover_repo_files(ROOT)
    finally:
        junk.unlink()
    assert "pyproject.toml" in found
    assert all(not path.startswith(OUTPUT_PREFIX) for path in found)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    missing = []
    for entry in tracked:
        if not entry:
            continue
        rel = entry.decode("utf-8").replace("\\", "/")
        if rel.startswith(OUTPUT_PREFIX):
            continue
        if rel not in found:
            missing.append(rel)
    assert missing == []


def test_model_init_depends_on_every_changed_byte() -> None:
    torch = pytest.importorskip("torch")
    from quant_fund.repo_llm.config import RepoLMConfig
    from quant_fund.repo_llm.model import RepoLM, fold_corpus_into_model

    cfg = RepoLMConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64, seq_len=16)
    tokens = np.arange(64, dtype=np.uint8)
    first = RepoLM(cfg)
    second = RepoLM(cfg)
    fold_corpus_into_model(first, tokens, seed=3)
    flipped = tokens.copy()
    flipped[-1] = np.uint8(int(flipped[-1]) ^ 1)
    fold_corpus_into_model(second, flipped, seed=3)
    left = torch.cat([param.detach().reshape(-1) for param in first.parameters()])
    right = torch.cat([param.detach().reshape(-1) for param in second.parameters()])
    assert not torch.equal(left, right)
    assert first.head.weight.data_ptr() == first.tok_emb.weight.data_ptr()


def test_attention_is_causal() -> None:
    torch = pytest.importorskip("torch")
    from quant_fund.repo_llm.config import RepoLMConfig
    from quant_fund.repo_llm.model import RepoLM

    torch.manual_seed(0)
    cfg = RepoLMConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, seq_len=16)
    model = RepoLM(cfg)
    left = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    right = left.clone()
    right[:, 10] = (right[:, 10] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_left, _ = model(left)
        logits_right, _ = model(right)
    assert torch.allclose(logits_left[:, :10], logits_right[:, :10])
    assert not torch.allclose(logits_left[:, 10], logits_right[:, 10])


def test_training_reduces_loss_and_resume_continues(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from quant_fund.repo_llm.config import RepoLMConfig
    from quant_fund.repo_llm.convert import convert_repository

    root = tmp_path / "repo"
    root.mkdir()
    text = ("abcdefg fedcba\n" * 80).encode()
    (root / "code.py").write_bytes(text * 4)
    (root / "note.md").write_bytes(b"train me\n" * 100)
    cfg = RepoLMConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, seq_len=32)
    out = tmp_path / "weights"
    files = ["code.py", "note.md"]
    first = convert_repository(
        root,
        out,
        steps=30,
        seed=5,
        batch_size=8,
        lr=3e-3,
        probe_batches=4,
        log_every=0,
        config=cfg,
        files=files,
    )
    assert first["probe_loss_final"] < first["probe_loss_initial"]
    assert first["step"] == 30
    assert first["sizes_the_book"] is False
    second = convert_repository(
        root,
        out,
        steps=2,
        seed=5,
        batch_size=8,
        lr=3e-3,
        resume=True,
        probe_batches=4,
        log_every=0,
        files=files,
    )
    assert second["step"] == 32
    assert second["corpus_sha256"] == first["corpus_sha256"]
    assert second["tokens_seen"] > first["tokens_seen"]
    assert second["probe_loss_initial"] == first["probe_loss_initial"]
