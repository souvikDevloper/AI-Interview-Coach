"""Validate a de-identified evidence CSV and emit its checksum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_interview_coach.evidence import read_csv_rows, sha256, validate_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("schema", choices=("retrieval", "wer", "misc", "survey"))
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    result = validate_rows(read_csv_rows(args.csv_file), args.schema)
    result["sha256"] = sha256(args.csv_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
