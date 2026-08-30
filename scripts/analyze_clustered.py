"""Run a cluster-level paired analysis on item-level evaluation CSV data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_interview_coach.analysis import cluster_paired_analysis
from ai_interview_coach.evidence import validate_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--treatment", default="D")
    parser.add_argument("--control", default="A")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--cluster-key", default="cluster_id")
    parser.add_argument("--condition-key", default="configuration")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows, "retrieval")
    metrics = args.metric or ["relevance", "star_score"]
    results = [
        cluster_paired_analysis(
            rows,
            metric=metric,
            treatment=args.treatment,
            control=args.control,
            cluster_key=args.cluster_key,
            condition_key=args.condition_key,
            bootstrap_replicates=args.bootstrap,
            permutation_replicates=args.permutations,
            seed=args.seed,
        ).to_dict()
        for metric in metrics
    ]
    payload = {
        "analysis": "cluster-level paired bootstrap and sign-flip randomisation test",
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
