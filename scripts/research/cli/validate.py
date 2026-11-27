"""Verify complete R1 run evidence without CUDA or simulator imports."""

import argparse
from pathlib import Path
import sys

from scripts.research.core.r1 import validate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        status = validate_run(Path(__file__).resolve().parents[3], args.run.resolve())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(f"PASS {status['valid_pairs']} pairs and {status['captured_observations_per_model']} observations per model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
