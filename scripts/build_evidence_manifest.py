"""Create a SHA-256 manifest for a versioned evidence archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_interview_coach.evidence import build_manifest, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evidence-manifest.json"))
    args = parser.parse_args()
    root = args.archive_root.resolve()
    output = args.output.resolve()
    paths = [path for path in root.rglob("*") if path.is_file() and path.resolve() != output]
    write_json(build_manifest(paths, root=root), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
