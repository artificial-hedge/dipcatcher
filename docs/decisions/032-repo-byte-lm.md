# ADR-032: Repository byte language model

## Status

Accepted

## Date

2026-09-23

## Context

ADR-007 keeps Torch optional and keeps neural nets out of the forecast stack
except for the lanes an ADR names. The repository itself can still be the
training corpus of a language model: source, docs, configs, and the smudged
data files. That artifact has to be trainable later, and it must not size
the book.

## Decision

1. `quant_fund.repo_llm` frames every git-tracked and unignored file
   into a lossless byte stream (`DCF1` length-prefixed frames). The only
   excluded tree is the converter output, `artifacts/repo_llm/`.
2. A decoder-only transformer is initialized by folding every corpus byte
   into its parameters (`corpus_fold_v1`), then trained with next-byte
   cross-entropy. `checkpoint.pt` stores the weights, AdamW state, corpus
   hash, and probe losses so `dipcatcher repo-llm --resume` can continue.
3. A corpus hash mismatch refuses to resume. Missing Torch fails when the
   command runs; importing the package does not import Torch.
4. This lane does not forecast returns and does not change `blend_weight`.

## Consequences

- `python -m quant_fund.repo_llm` and `dipcatcher repo-llm` write
  `artifacts/repo_llm/`. `corpus.bin` is gitignored and rebuilt from the
  working tree. Git LFS files are included as smudged bytes when present.
- ADR-007 stays in force for every other neural architecture.
