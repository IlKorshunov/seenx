
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_one_partial(partial_path: Path, delete_partial: bool = False) -> bool:
    name = partial_path.name
    if not name.endswith("_features.csv.partial"):
        return False
    final_path = partial_path.parent / name[: -len(".partial")]
    part = pd.read_csv(partial_path, index_col=0)
    if part.index.name is None and len(part.columns):
        pass

    if final_path.exists():
        main = pd.read_csv(final_path, index_col=0)
        idx = part.index.union(main.index).sort_values()
        combined = pd.DataFrame(index=idx)
        for c in main.columns:
            combined[c] = main[c].reindex(combined.index)
        for c in part.columns:
            combined[c] = part[c].reindex(combined.index)
        combined.index.name = main.index.name or part.index.name
    else:
        combined = part.copy()

    combined.to_csv(final_path, index=True)
    if delete_partial:
        partial_path.unlink(missing_ok=False)
    print(f"[merge] {partial_path.name} -> {final_path.name} ({len(combined.columns)} cols)")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Merge *_features.csv.partial into *_features.csv")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Каталог с CSV (рекурсивно ищем *_features.csv.partial)")
    p.add_argument("--delete-partial", action="store_true", help="После merge удалить .partial (по умолчанию не трогаем)")
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
            if merge_one_partial(path, delete_partial=args.delete_partial):
                n += 1
        except Exception as e:
            print(f"failed: {path}; {e}")
    print(f"done: {n} files")


if __name__ == "__main__":
    main()
