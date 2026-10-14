"""
prepare_external.py - Prepare the IDRiD disease-grading images as an EXTERNAL test set.

PURPOSE
    Convert the IDRiD images (516 photographs from an eye clinic in Nanded, India)
    into the same format as our APTOS pipeline, so the trained model can be tested
    on images it has never seen. IDRiD is used for evaluation ONLY: never for
    training, epoch selection or threshold tuning.

STEPS
    1. read the two ground-truth CSV files (training and testing labels)
    2. check that every labelled image exists
    3. apply EXACTLY the same preprocessing as APTOS (src.preprocess.preprocess:
       crop black borders, resize to 512x512, CLAHE on the L channel)
    4. save one PNG per image and a labels file

INPUTS (unzipped IEEE DataPort archive)
    data/external/idrid/raw/B. Disease Grading/1. Original Images/a. Training Set/*.jpg
    data/external/idrid/raw/B. Disease Grading/1. Original Images/b. Testing Set/*.jpg
    data/external/idrid/raw/B. Disease Grading/2. Groundtruths/*.csv

OUTPUTS
    data/external/idrid/processed_512/<split>_<name>.png   e.g. train_IDRiD_001.png
    data/external/idrid/labels.csv    columns: id_code, diagnosis, split, orig_name
                                      (id_code/diagnosis match the APTOS format, so
                                      src.dataset.RetinaDataset works unchanged)

NOTES
    - The split prefix avoids name collisions between the training and testing sets.
    - Runs are resumable: existing PNG files are skipped.
    - IDRiD is licensed CC-BY-4.0: cite the dataset (Porwal et al., Data 2018, 3(3), 25).

USAGE (from the project root, with the virtual environment active)
    python -m src.prepare_external
"""

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.preprocess import preprocess

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "external" / "idrid"
RAW = BASE / "raw" / "B. Disease Grading"
OUT_IMG = BASE / "processed_512"
IMG_FOLDERS = {"train": "a. Training Set", "test": "b. Testing Set"}


def split_of(csv_path):
    """Decide from the file name whether a label CSV belongs to the train or test set."""
    name = csv_path.name.lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    raise ValueError(f"Cannot tell if '{csv_path.name}' is a training or a testing file")


def find_columns(df, csv_name):
    """Return (image-name column, retinopathy-grade column) of a ground-truth CSV."""
    cols = list(df.columns)
    id_col = next((c for c in cols if "image" in str(c).lower()), None)
    grade_col = next((c for c in cols if "retinopathy" in str(c).lower()), None)
    if id_col is None or grade_col is None:
        raise ValueError(f"Cannot find the image-name and retinopathy-grade columns in "
                         f"{csv_name}. Columns found: {cols}")
    return id_col, grade_col


def main():
    """Prepare all labelled IDRiD images and write labels.csv."""
    gt_dir = RAW / "2. Groundtruths"
    csvs = sorted(gt_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV file found in {gt_dir}")

    # 1. Collect (split, name, path, grade) for every labelled image
    items, missing = [], []
    for csv_path in csvs:
        split = split_of(csv_path)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        id_col, grade_col = find_columns(df, csv_path.name)
        df = df[[id_col, grade_col]].dropna()
        print(f"{csv_path.name}: {len(df)} rows | image column '{id_col}' | grade column '{grade_col}'")

        img_dir = RAW / "1. Original Images" / IMG_FOLDERS[split]
        files = {p.stem: p for p in img_dir.iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        for _, r in df.iterrows():
            name = Path(str(r[id_col]).strip()).stem
            grade = int(r[grade_col])
            assert 0 <= grade <= 4, f"Unexpected grade {grade} for {name}"
            if name in files:
                items.append((split, name, files[name], grade))
            else:
                missing.append(f"{split}/{name}")

    if missing:
        raise FileNotFoundError(f"{len(missing)} labelled images have no file, e.g. {missing[:5]}")
    ids = [f"{s}_{n}" for s, n, _, _ in items]
    assert len(ids) == len(set(ids)), "Duplicate image ids after adding the split prefix"

    # 2. Preprocess (same function as APTOS)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    for (split, name, path, grade), code in tqdm(list(zip(items, ids))):
        dst = OUT_IMG / f"{code}.png"
        if dst.exists():
            continue
        img = cv2.imread(str(path))
        assert img is not None, f"Could not read {path}"
        img = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 3. Labels file
    labels = pd.DataFrame({"id_code": ids,
                           "diagnosis": [g for _, _, _, g in items],
                           "split": [s for s, _, _, _ in items],
                           "orig_name": [n for _, n, _, _ in items]})
    labels.to_csv(BASE / "labels.csv", index=False)

    print(f"\nImages prepared: {len(labels)}  (expected 516)")
    print(pd.crosstab(labels["diagnosis"], labels["split"], margins=True).to_string())
    n_files = len(list(OUT_IMG.glob("*.png")))
    print(f"PNG files in {OUT_IMG.name}: {n_files}")


if __name__ == "__main__":
    main()