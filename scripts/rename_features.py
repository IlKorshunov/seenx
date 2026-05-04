"""Rename old feature columns in output CSVs to match current extractors."""

import glob

import pandas as pd


RENAME = {"filler_density": "crutch_cnt", "sentiment_polarity": "pos_cnt", "sentiment_intensity": "total_emotional_cnt"}

DROP = {"question_density", "topic_shift", "frame", "n_ad_segments", "ad_density_percent", "ad_density_pct", "cultural_ref_density", "cultural_ref_cnt"}


def migrate(path: str) -> None:
    df = pd.read_csv(path, index_col=0)
    cols_before = set(df.columns)
    changed = False

    for old, new in RENAME.items():
        if old not in df.columns:
            continue
        if new in df.columns:
            df.drop(columns=[old], inplace=True)
        else:
            df.rename(columns={old: new}, inplace=True)
        changed = True

    to_drop = DROP & set(df.columns)
    if to_drop:
        df.drop(columns=list(to_drop), inplace=True)
        changed = True

    if changed:
        removed = cols_before - set(df.columns)
        renamed = {o: n for o, n in RENAME.items() if o in cols_before and n not in cols_before}
        df.to_csv(path, index=True)
        print(f"  {path}")
        if renamed:
            print(f"    renamed: {renamed}")
        if removed - set(renamed):
            print(f"    dropped: {removed - set(renamed)}")
    else:
        print(f"  {path}  (no changes)")


if __name__ == "__main__":
    files = sorted(glob.glob("output/*_features.csv"))
    print(f"Found {len(files)} feature CSVs\n")
    for f in files:
        migrate(f)
    print("\nDone.")
