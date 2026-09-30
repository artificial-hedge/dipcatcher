from pathlib import Path
p = Path('tests/unit/test_sota_finalization_verifier.py')
s = p.read_text()
s = s.replace('run = repo / "runs" / "valid-run"', 'run = inp / "runs" / "valid-run"')
p.write_text(s)
