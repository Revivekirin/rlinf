"""Validate sealed R2-A unit coverage and comparisons on CPU."""

import argparse
import json
from pathlib import Path
import sys

from scripts.research.experiments.module_rollback import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--r1", required=True, type=Path)
    args = parser.parse_args()
    try:
        status = validate(Path(__file__).resolve().parents[3], args.run.resolve(), args.r1.resolve())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2))
    return 2 if status["status"] == "PARTIAL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
