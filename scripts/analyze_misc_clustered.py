"""Analyse MI-consistent/inconsistent rates at the independent session level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_interview_coach.analysis import cluster_independent_analysis
from ai_interview_coach.evidence import read_csv_rows, validate_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_csv_rows(args.input_csv)
    validate_rows(rows, "misc")
    results = [
        cluster_independent_analysis(
            rows,
            metric=metric,
            treatment=args.treatment,
            control=args.control,
            cluster_key="session_id",
            condition_key="system",
            bootstrap_replicates=args.bootstrap,
            permutation_replicates=args.permutations,
            seed=args.seed,
        ).to_dict()
        for metric in ("mi_consistent", "mi_inconsistent")
    ]
    payload = {
        "analysis": "session-level independent cluster bootstrap and label-permutation test",
        "input": str(args.input_csv),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
