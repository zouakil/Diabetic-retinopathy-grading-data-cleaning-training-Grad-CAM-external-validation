"""
clean_data.py - Audit and clean the raw APTOS 2019 images before any preprocessing.

PURPOSE
    Inspect every raw image once, remove the clearly unusable ones, remove
    duplicate images, and flag (but do not remove) unusual-looking images for
    manual review. Everything downstream (preprocessing, splitting, training)
    then starts from the cleaned list instead of the raw dataset.

WHY THIS STEP EXISTS
    APTOS contains exact duplicate photos, and some duplicates carry different
    grades. If copies end up on both sides of a train/validation split, the
    model has already seen part of the validation set, which inflates the scores.

INPUTS
    data/train.csv              columns: id_code, diagnosis
    data/train_images/*.png     the raw images
    data/manual_exclude.txt     OPTIONAL, one id_code per line, filled after the
                                visual review of the flagged images

OUTPUTS
    data/image_report.csv       one row per image: measurements, the reason it was
                                removed (if any), and the review flags
    data/clean.csv              the final list of kept images (id_code, diagnosis)

RULES, APPLIED IN THIS ORDER
    1. REMOVE  unreadable files          OpenCV cannot open them
    2. REMOVE  images that are too small shorter side < MIN_SIDE pixels
    3. REMOVE  empty / black images      fewer than EMPTY_FRAC of pixels are non-black
    4. REMOVE  every copy of a duplicate group whose copies have different grades
               (we cannot know which label is right)
    5. REMOVE  extra copies of a duplicate group whose grades agree
               (one copy is kept, the first by id_code)
    6. REMOVE  images listed in data/manual_exclude.txt
    7. FLAG    unusually dark, bright, blurry or low-retina images among the kept
               ones (the lowest or highest 1%). Flagged images are NOT removed,
               because deleting hard cases would make the results look better
               than they really are. Review them visually first.

LIMITATIONS
    Only EXACT duplicates are detected (identical pixels). The same eye
    photographed twice, or a resized copy, is not detected.

USAGE (from the project root, with the virtual environment active)
    python -m src.clean_data
"""

import hashlib
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.preprocess import crop_black_borders

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MIN_SIDE = 256        # images with a shorter side than this are removed
EMPTY_FRAC = 0.05     # fewer than 5% non-black pixels = empty image


def inspect_image(path):
    """Measure one image and return a dict of its properties.

    Returns: readable, width, height, retina_frac (share of non-black pixels),
    brightness, blur (low = blurry) and hash (fingerprint used to find
    duplicates). If the file cannot be opened, returns early with
    readable=False so that one bad file never stops the scan.
    """
    info = {"readable": False, "width": 0, "height": 0,
            "retina_frac": 0.0, "brightness": 0.0, "blur": 0.0, "hash": None}

    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return info

    info["readable"] = True
    h, w = img.shape[:2]
    info["width"], info["height"] = w, h

    # Exact-duplicate fingerprint: same pixels (and same shape) = same hash
    info["hash"] = hashlib.md5(img.tobytes() + str(img.shape).encode()).hexdigest()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    info["retina_frac"] = float((gray > 7).mean())          # share of non-black pixels

    # Brightness and sharpness on a standard 512x512 crop, so images of
    # different resolutions are comparable.
    small = cv2.resize(crop_black_borders(rgb), (512, 512), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    info["brightness"] = float(g.mean())
    info["blur"] = float(cv2.Laplacian(g, cv2.CV_64F).var())  # low = blurry
    return info


def main():
    """Run the full audit: measure, remove, flag, then write the two CSV files."""
    df = pd.read_csv(DATA / "train.csv")
    rows = [inspect_image(DATA / "train_images" / f"{c}.png") for c in tqdm(df["id_code"])]
    rep = pd.concat([df, pd.DataFrame(rows)], axis=1)

    # 1. Clearly unusable images
    rep["bad_unreadable"] = ~rep["readable"]
    rep["bad_small"] = rep["readable"] & (rep[["width", "height"]].min(axis=1) < MIN_SIDE)
    rep["bad_empty"] = rep["readable"] & (rep["retina_frac"] < EMPTY_FRAC)
    basic_bad = rep.bad_unreadable | rep.bad_small | rep.bad_empty

    # 2. Duplicates: drop every copy if the labels disagree, otherwise keep one
    ok = rep[~basic_bad]
    grp = ok.groupby("hash")["diagnosis"].agg(n="size", n_labels="nunique")
    conflict_hashes = set(grp[(grp.n > 1) & (grp.n_labels > 1)].index)
    rep["dup_conflict"] = (~basic_bad) & rep["hash"].isin(conflict_hashes)

    cand = rep[~basic_bad & ~rep.dup_conflict].sort_values("id_code")
    extra = cand.index[cand.duplicated("hash", keep="first")]
    rep["dup_extra"] = rep.index.isin(extra)

    # 3. Optional manual exclusions (one id_code per line), filled after visual review
    manual_file = DATA / "manual_exclude.txt"
    manual = set()
    if manual_file.exists():
        manual = {l.strip() for l in manual_file.read_text().splitlines() if l.strip()}
    rep["manual_exclude"] = rep["id_code"].isin(manual)

    rep["keep"] = ~(basic_bad | rep.dup_conflict | rep.dup_extra | rep.manual_exclude)

    # 4. Outlier flags among kept images (flag only, nothing is removed)
    k = rep[rep["keep"]]
    p = lambda col, q: k[col].quantile(q)
    rep["flag_dark"] = rep["keep"] & (rep["brightness"] < p("brightness", 0.01))
    rep["flag_bright"] = rep["keep"] & (rep["brightness"] > p("brightness", 0.99))
    rep["flag_blurry"] = rep["keep"] & (rep["blur"] < p("blur", 0.01))
    rep["flag_low_retina"] = rep["keep"] & (rep["retina_frac"] < p("retina_frac", 0.01))

    rep.to_csv(DATA / "image_report.csv", index=False)
    rep.loc[rep["keep"], ["id_code", "diagnosis"]].to_csv(DATA / "clean.csv", index=False)

    print("\n=== Cleaning summary ===")
    print("Total images:                      ", len(rep))
    print("Unreadable:                        ", int(rep.bad_unreadable.sum()))
    print("Too small:                         ", int(rep.bad_small.sum()))
    print("Empty / black:                     ", int(rep.bad_empty.sum()))
    print("Duplicates, conflicting labels:    ", int(rep.dup_conflict.sum()))
    print("Duplicates, extra copies removed:  ", int(rep.dup_extra.sum()))
    print("Manual exclusions:                 ", int(rep.manual_exclude.sum()))
    print("KEPT:                              ", int(rep["keep"].sum()))
    print("Kept per grade:")
    print(rep.loc[rep["keep"], "diagnosis"].value_counts().sort_index().to_string())
    print("Flagged for review (not removed):  ",
          int(rep[["flag_dark", "flag_bright", "flag_blurry", "flag_low_retina"]].any(axis=1).sum()))


if __name__ == "__main__":
    main()