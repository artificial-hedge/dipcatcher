from pathlib import Path

# Keep the finalizer's manifest self-describing: each group points to its durable command record.
p = Path('scripts/finalize_sota.ps1')
s = p.read_text()
old = '''   $record = [ordered]@{ group=$group.name; command=@($Python)+$args; command_text=$commandText; cwd=$repo; shards=$shards; stdout=$stdout; stderr=$stderr; transcript=$transcript; receipt=$output; started_utc=(Get-Date).ToUniversalTime().ToString('o') }
'''
new = '''   $commandRecord = Join-Path $runRoot ("{0}.command.json" -f $group.name)
   $record = [ordered]@{ group=$group.name; command=@($Python)+$args; command_text=$commandText; cwd=$repo; shards=$shards; stdout=$stdout; stderr=$stderr; transcript=$transcript; receipt=$output; command_record=$commandRecord; started_utc=(Get-Date).ToUniversalTime().ToString('o') }
'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace("Write-Json (Join-Path $runRoot (\"{0}.command.json\" -f $group.name)) $record", "Write-Json $commandRecord $record")
p.write_text(s)

p = Path('tests/unit/test_sota_finalization_verifier.py')
s = p.read_text()
s = s.replace('''    stdout, stderr, transcript = (run / n for n in (
        "group.stdout.txt", "group.stderr.txt", "group.transcript.txt"))
''', '''    stdout, stderr, transcript = (run / n for n in (
        "group.stdout.txt", "group.stderr.txt", "group.transcript.txt"))
    command_record = run / "group.command.json"
''')
s = s.replace('''    group = {
        "group": "fixture", "command": ["python", "evaluator"],
''', '''    group = {
        "group": "fixture", "command": ["python", "evaluator"],
''')
old = '''        "receipt": str(receipt), "receipt_sha256": sha(receipt),
        "losses": str(losses), "losses_sha256": sha(losses),
        "exit_code": 0, "status": "succeeded",
    }
'''
new = '''        "receipt": str(receipt), "receipt_sha256": sha(receipt),
        "losses": str(losses), "losses_sha256": sha(losses),
        "command_record": str(command_record),
        "exit_code": 0, "status": "succeeded",
    }
    command_record.write_text(json.dumps(group))
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
