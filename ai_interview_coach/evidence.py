"""Validation and checksums for de-identified item-level evidence archives."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


SCHEMAS = {
    "retrieval": {
        "required": {"cluster_id", "item_id", "configuration", "mode", "relevance", "star_score"},
        "ranges": {"relevance": (1.0, 5.0), "star_score": (0.0, 100.0)},
        "unique": ("cluster_id", "item_id", "configuration"),
    },
    "wer": {
        "required": {"session_id", "utterance_id", "condition", "reference", "hypothesis"},
        "ranges": {},
        "unique": ("session_id", "utterance_id"),
    },
    "misc": {
        "required": {
            "session_id", "utterance_id", "coder_id", "system", "misc_code",
            "mi_consistent", "mi_inconsistent",
        },
        "ranges": {"mi_consistent": (0.0, 1.0), "mi_inconsistent": (0.0, 1.0)},
        "unique": ("session_id", "utterance_id", "coder_id"),
    },
    "survey": {
        "required": {"participant_id", "institution_id", "item_id", "response"},
        "ranges": {"response": (1.0, 5.0)},
        "unique": ("participant_id", "item_id"),
    },
}

BANNED_IDENTIFIER_COLUMNS = {
    "name", "full_name", "email", "phone", "address", "student_email", "student_name", "roll_number"
}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_rows(rows: Iterable[dict[str, object]], schema_name: str) -> dict[str, object]:
    if schema_name not in SCHEMAS:
        raise ValueError(f"unknown schema {schema_name!r}")
    records = list(rows)
    if not records:
        raise ValueError("evidence file contains no data rows")
    schema = SCHEMAS[schema_name]
    columns = set(records[0])
    banned = columns & BANNED_IDENTIFIER_COLUMNS
    if banned:
        raise ValueError(f"direct identifier columns are not permitted: {sorted(banned)}")
    missing = schema["required"] - columns
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    seen: set[tuple[str, ...]] = set()
    for line_number, row in enumerate(records, start=2):
        key = tuple(str(row.get(column, "")).strip() for column in schema["unique"])
        if not all(key):
            raise ValueError(f"blank identifier at CSV line {line_number}")
        if key in seen:
            raise ValueError(f"duplicate evidence key {key} at CSV line {line_number}")
        seen.add(key)
        for column, (low, high) in schema["ranges"].items():
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column} must be numeric at CSV line {line_number}") from exc
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{column}={value!r} is outside [{low}, {high}] at CSV line {line_number}")
        if schema_name == "misc":
            consistent = float(row["mi_consistent"])
            inconsistent = float(row["mi_inconsistent"])
            if consistent not in {0.0, 1.0} or inconsistent not in {0.0, 1.0}:
                raise ValueError(f"MISC indicators must be binary at CSV line {line_number}")
            if consistent + inconsistent > 1.0:
                raise ValueError(f"an utterance cannot be both MI-consistent and MI-inconsistent at CSV line {line_number}")
    return {"schema": schema_name, "rows": len(records), "columns": sorted(columns)}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(paths: Iterable[str | Path], *, root: str | Path) -> dict[str, object]:
    root_path = Path(root).resolve()
    files = []
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        if root_path not in path.parents and path != root_path:
            raise ValueError(f"manifest path is outside archive root: {path}")
        files.append({
            "path": path.relative_to(root_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return {"manifest_version": 1, "files": files}


def write_json(data: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
