"""Repository byte language model.

The training corpus is the git working tree. A decoder-only transformer is
initialized from every corpus byte and trained with next-byte loss. Importing
this package does not import torch; training lives in ``convert``.
"""

from quant_fund.repo_llm.config import RepoLMConfig
from quant_fund.repo_llm.corpus import (
    CORPUS_FILENAME,
    MAGIC,
    MANIFEST_FILENAME,
    OUTPUT_PREFIX,
    VOCAB_SIZE,
    CorpusManifest,
    RepoLLMError,
    build_corpus,
    discover_repo_files,
    ensure_corpus,
    load_manifest,
    verify_corpus,
)
from quant_fund.repo_llm.fold import fold_lanes

__all__ = [
    "CORPUS_FILENAME",
    "MAGIC",
    "MANIFEST_FILENAME",
    "OUTPUT_PREFIX",
    "VOCAB_SIZE",
    "CorpusManifest",
    "RepoLMConfig",
    "RepoLLMError",
    "build_corpus",
    "discover_repo_files",
    "ensure_corpus",
    "fold_lanes",
    "load_manifest",
    "verify_corpus",
]
