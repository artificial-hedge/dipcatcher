import json
from pathlib import Path
for f in sorted(Path(r"D:\dipcatcher\.dsh-24x7\eval-full").glob("s23_*.json")):
    d = json.load(open(f))
    keys = list(d.keys())
    scalar = {k: d[k] for k in keys if not isinstance(d[k], (dict, list))}
    print(f.name)
    print("  scalars:", scalar)
    for k in keys:
        if isinstance(d[k], dict):
            print(f"  dict[{k}]:", list(d[k].keys())[:8])
        elif isinstance(d[k], list):
            print(f"  list[{k}]: n={len(d[k])}")
