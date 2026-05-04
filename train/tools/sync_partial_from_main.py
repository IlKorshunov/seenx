#!/usr/bin/env python3
"""
Дописывает в `*_features.csv.partial` колонки из основного `*_features.csv`,
если в основном больше признаков; основной файл не изменяется.

Обратное направление к merge_output_partials.py (там partial -> main по колонкам).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sync_one_partial(partial_path: Path, dry_run: bool = False) -> bool:
    name = partial_path.name
    if not name.endswith("_features.csv.partial"):
        return False
    main_path = partial_path.parent / name[: -len(".partial")]
    if not main_path.exists():
        print(f"[skip] no main file: {main_path.name}")
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

    if dry_run:
        print(f"[dry-run] would add {len(extra_cols)} col(s) to {partial_path.name}: {list(extra_cols)[:8]}{'...' if len(extra_cols) > 8 else ''}")
        return True

    combined.to_csv(partial_path, index=True)
    print(f"[sync] {main_path.name} -> {partial_path.name} (+{len(extra_cols)} cols, now {len(combined.columns)} cols)")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Copy missing feature columns from *_features.csv into *_features.csv.partial")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Каталог с CSV (рекурсивно ищем *_features.csv.partial)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = args.output_dir
    if not root.is_dir():
        print(f"[skip] output dir missing: {root}")
        return
    partials = sorted(root.rglob("*_features.csv.partial"))
    if not partials:
        print(f"[sync] no partial files under {root}")
        return
    n = 0
    for path in partials:
        try:
            if sync_one_partial(path, dry_run=args.dry_run):
                n += 1
        except Exception as e:
            print(f"[fail] {path}: {e}")
    print(f"[sync] done: updated {n} file(s)")


if __name__ == "__main__":
    main()
