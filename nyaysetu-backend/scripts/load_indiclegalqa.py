"""Load the IndicLegalQA dataset and print the first 5 rows.

Usage:
    python scripts/load_indiclegalqa.py
    python scripts/load_indiclegalqa.py --dir data/indiclegalqa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_data_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    extensions = {".csv", ".json", ".jsonl", ".parquet", ".tsv"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def _load(path: Path):
    """Return a pandas DataFrame for the file."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        # Try records-style JSON; fall back to a single-object load.
        try:
            return pd.read_json(path)
        except ValueError:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                # Common HF-style: {"data": [...]} or nested
                for key in ("data", "rows", "records"):
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
    raise ValueError(f"Unsupported file format: {path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="data/indiclegalqa",
        help="Directory containing the unzipped dataset (default: data/indiclegalqa)",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    files = _find_data_files(root)
    if not files:
        print(f"No data files found in {root}.", file=sys.stderr)
        print("Make sure you have downloaded and unzipped the dataset first.", file=sys.stderr)
        return 1

    print(f"Found {len(files)} data file(s) in {root}:")
    for f in files:
        print(f"  - {f.relative_to(root)}")
    print()

    target = files[0]
    print(f"Loading: {target.name}")
    df = _load(target)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("First 5 rows:")
    print(df.head().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
