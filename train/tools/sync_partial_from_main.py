
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sync_one_partial(partial_path: Path) -> bool:
    name = partial_path.name
    if not name.endswith("_features.csv.partial"):
        return False
    main_path = partial_path.parent / name[: -len(".partial")]
    if not main_path.exists():
        print(f"no file: {main_path.name}")
        return False

    main = pd.read_csv(main_path, index_col=0)
    part = pd.read_csv(partial_path, index_col=0)
    extra_cols = main.columns.difference(part.columns)
    if len(extra_cols) == 0:
        return False

    combined = part.copy()
    for c in extra_cols:
        combined[c] = main[c].reindex(combined.index)
    combined.index.name = part.index.name or main.index.name

    combined.to_csv(partial_path, index=True)
    print(f"{partial_path.name}: +{list(extra_cols)}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Copy missing feature columns from *_features.csv into *_features.csv.partial")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Каталог с CSV (рекурсивно ищем *_features.csv.partial)")
    args = p.parse_args()
    root = args.output_dir
    if not root.is_dir():
        print(f"doesn't exist: {root}")
        return
    partials = sorted(root.rglob("*_features.csv.partial"))
    if not partials:
        print(f"no partial files {root}")
        return
    n = 0
    for path in partials:
        try:
            if sync_one_partial(path):
                n += 1
        except Exception as e:
            print(f"failed: {path}; {e}")
    print(f"done: {n} files")


if __name__ == "__main__":
    main()
