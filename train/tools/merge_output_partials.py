#!/usr/bin/env python3
"""
Слияние checkpoint-файлов `*_features.csv.partial` (новые колонки из агрегатора)
в основной `*_features.csv`, чтобы пайплайн обучения их подхватывал.

Файл `.partial` по умолчанию сохраняется; удалить можно флагом `--delete-partial`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_one_partial(partial_path: Path, dry_run: bool = False, delete_partial: bool = False) -> bool:
    name = partial_path.name
    if not name.endswith("_features.csv.partial"):
        return False
    final_path = partial_path.parent / name[: -len(".partial")]
    part = pd.read_csv(partial_path, index_col=0)
    if part.index.name is None and len(part.columns):
        # часто индекс — time
        pass

    if final_path.exists():
        main = pd.read_csv(final_path, index_col=0)
        idx = part.index.union(main.index)
        try:
            idx = idx.sort_values()
        except Exception:
            idx = pd.Index(sorted(idx.unique()))
        combined = pd.DataFrame(index=idx)
        for c in main.columns:
            combined[c] = main[c].reindex(combined.index)
        for c in part.columns:
            combined[c] = part[c].reindex(combined.index)
        combined.index.name = main.index.name or part.index.name
    else:
        combined = part.copy()

    if dry_run:
        print(f"[dry-run] would write {final_path} ({len(combined)} rows, {len(combined.columns)} cols)")
        return True

    combined.to_csv(final_path, index=True)
    if delete_partial:
        partial_path.unlink(missing_ok=False)
    print(f"[merge] {partial_path.name} -> {final_path.name} ({len(combined.columns)} cols)")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Merge *_features.csv.partial into *_features.csv")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Каталог с CSV (рекурсивно ищем *_features.csv.partial)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delete-partial", action="store_true", help="После merge удалить .partial (по умолчанию не трогаем)")
    args = p.parse_args()
    root = args.output_dir
    if not root.is_dir():
        print(f"[skip] output dir missing: {root}")
        return
    partials = sorted(root.rglob("*_features.csv.partial"))
    if not partials:
        print(f"[merge] no partial files under {root}")
        return
    n = 0
    for path in partials:
        try:
            if merge_one_partial(path, dry_run=args.dry_run, delete_partial=args.delete_partial):
                n += 1
        except Exception as e:
            print(f"[fail] {path}: {e}")
    print(f"[merge] done: {n} file(s)")


if __name__ == "__main__":
    main()
