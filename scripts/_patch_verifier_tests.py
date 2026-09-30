from pathlib import Path
p = Path('tests/unit/test_sota_finalization_verifier.py')
s = p.read_text()
s = s.replace('"source_parts_sha256": {shard.name: sha(shard)},', '"source_parts_sha256": {"eval/part.losses.npz": sha(shard)},')
p.write_text(s)
