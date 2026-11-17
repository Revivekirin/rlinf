"""Phase 3A/3B JSON and CSV serialization contracts."""

import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def csv_save(path, rows):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        if rows:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
