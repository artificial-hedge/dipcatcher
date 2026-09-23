Dipcatcher repository byte language model

This checkpoint was initialized by folding every byte of the repository
corpus into the transformer weights, then trained with next-byte
cross-entropy. It is not a forecast and it does not size the book.

files: 2588
corpus_bytes: 1966540994
corpus_sha256: f93ac2279afed2d399bab28759017e3f790d4b21a7fd96454111043aa156da65
parameters: 4820224
optimizer_steps: 10202
probe_loss_initial: 5.600858747959137
probe_loss_final: 5.257237523794174

Continue training from this directory:
  python -m quant_fund.repo_llm --resume --steps 500
  dipcatcher repo-llm --resume --steps 500

Requires the optional torch extra: pip install 'dipcatcher[nn]'.
Excluded from the corpus: artifacts/repo_llm/ (this output).
