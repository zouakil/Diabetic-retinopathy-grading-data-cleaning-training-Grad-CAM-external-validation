"""
make_split.py - Create the single train/validation split from the cleaned dataset.

PURPOSE
    Split data/clean.csv once into 80% training and 20% validation, and save the
    two lists so every later notebook and script uses exactly the same images.
    Never re-split later: a new split could put former validation images into
    training and invalidate earlier results.

DESIGN CHOICES
    - Stratified on the grade: both sets keep the same proportion of each grade
      (important because grades 3 and 4 are rare).
    - Fixed seed (42): the split is reproducible.
    - Duplicates were removed beforehand by src/clean_data.py, so a plain
      stratified split cannot put a copy of an image on both sides. This script
      still VERIFIES it (by id and by image hash) and stops with an error if
      the check fails.

INPUTS
    data/clean.csv           from src/clean_data.py
    data/image_report.csv    from src/clean_data.py (used for the hash check)

OUTPUTS
    data/splits/train.csv    columns: id_code, diagnosis
    data/splits/val.csv      columns: id_code, diagnosis

USAGE (from the project root, with the virtual environment active)
    python -m src.make_split
    Re-running refuses to overwrite an existing split unless --force is given.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
VAL_SIZE = 0.2


def main():
    clean = pd.read_csv(DATA / "clean.csv")
    report = pd.read_csv(DATA / "image_report.csv")[["id_code", "hash"]]

    train, val = train_test_split(
        clean, test_size=VAL_SIZE, stratify=clean["diagnosis"], random_state=SEED
    )

    # --- Integrity checks: the script stops if any of them fails ---
    ids_overlap = set(train["id_code"]) & set(val["id_code"])
    h = report.set_index("id_code")["hash"]
    hash_overlap = set(h[train["id_code"]]) & set(h[val["id_code"]])
    assert len(ids_overlap) == 0, f"{len(ids_overlap)} images are in both sets"
    assert len(hash_overlap) == 0, f"{len(hash_overlap)} identical images are in both sets"

    out = DATA / "splits"
    if (out / "train.csv").exists() and "--force" not in sys.argv:
        raise SystemExit("data/splits already exists. Refusing to overwrite it "
                         "(re-splitting would invalidate earlier results). "
                         "Use --force only if you really mean it.")
    out.mkdir(parents=True, exist_ok=True)
    train.sort_values("id_code").to_csv(out / "train.csv", index=False)
    val.sort_values("id_code").to_csv(out / "val.csv", index=False)

    print(f"Train: {len(train)} | Val: {len(val)}")
    print("Overlap by id: 0 | Overlap by image hash: 0  (checks passed)")
    print("\nImages per grade:")
    print(pd.DataFrame({
        "train": train["diagnosis"].value_counts().sort_index(),
        "val": val["diagnosis"].value_counts().sort_index(),
    }).to_string())


if __name__ == "__main__":
    main()