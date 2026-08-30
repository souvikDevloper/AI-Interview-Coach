"""Validate the versioned LoRA configuration and approved corpus contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {"pair_id", "instruction", "input", "output", "split"}


def load_records(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
        records.append(record)
    if not records:
        raise ValueError("training JSONL has no records")
    if len({str(record["pair_id"]) for record in records}) != len(records):
        raise ValueError("pair_id values must be unique")
    return records


def validate_splits(records: list[dict[str, object]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        split = str(record["split"])
        if split not in counts:
            raise ValueError(f"unsupported split {split!r}")
        counts[split] += 1
    if not all(counts.values()):
        raise ValueError("train, validation, and test splits must all be non-empty")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_jsonl", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/lora_llama2_7b.json"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = load_records(args.data_jsonl)
    counts = validate_splits(records)
    print(json.dumps({"config": config, "split_counts": counts}, indent=2, sort_keys=True))
    if args.validate_only:
        return 0
    raise SystemExit(
        "Training remains gated until the approved de-identified corpus is supplied. "
        "Record the exact package lock, hardware, seed, and output checksums when running PEFT/TRL."
    )


if __name__ == "__main__":
    raise SystemExit(main())
