"""Calculate micro and session-macro WER without exposing raw audio."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_interview_coach.evidence import read_csv_rows, validate_rows
from ai_interview_coach.metrics import word_error_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_csv_rows(args.input_csv)
    validate_rows(rows, "wer")
    by_condition: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_session: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        errors, words = word_error_counts(row["reference"], row["hypothesis"])
        by_condition[row["condition"]].append((errors, words))
        totals = by_session[(row["condition"], row["session_id"])]
        totals[0] += errors
        totals[1] += words
    conditions = []
    for condition in sorted(by_condition):
        pairs = by_condition[condition]
        errors = sum(pair[0] for pair in pairs)
        words = sum(pair[1] for pair in pairs)
        session_wers = [
            total[0] / total[1]
            for (group, _session), total in by_session.items()
            if group == condition and total[1]
        ]
        conditions.append({
            "condition": condition,
            "utterances": len(pairs),
            "sessions": len(session_wers),
            "reference_words": words,
            "micro_wer": errors / words if words else None,
            "session_macro_wer": sum(session_wers) / len(session_wers) if session_wers else None,
        })
    rendered = json.dumps({"input": str(args.input_csv), "conditions": conditions}, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
