"""Validate survey rows and report reliability plus item/group descriptives."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_interview_coach.evidence import read_csv_rows, validate_rows
from ai_interview_coach.metrics import cronbach_alpha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_csv_rows(args.input_csv)
    validate_rows(rows, "survey")
    by_item: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(float(row["response"]))
    payload: dict[str, object] = {
        "input": str(args.input_csv),
        "participants": len({row["participant_id"] for row in rows}),
        "items": len(by_item),
        "cronbach_alpha": cronbach_alpha(rows),
        "item_means": {item: statistics.fmean(values) for item, values in sorted(by_item.items())},
    }
    grouped: dict[str, object] = {}
    for column in args.group:
        if column not in rows[0]:
            raise ValueError(f"group column is missing: {column}")
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            buckets[row[column]].append(float(row["response"]))
        grouped[column] = {
            value: {"responses": len(values), "mean": statistics.fmean(values)}
            for value, values in sorted(buckets.items())
        }
    payload["group_descriptives"] = grouped
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
