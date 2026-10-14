"""
preprocess.py - Crop, resize and enhance the cleaned retinal images.

PURPOSE
    Turn every KEPT raw image (listed in data/clean.csv) into a standardized
    512x512 image, saved once to disk so training does not repeat this work
    at every epoch.

STEPS (applied to each image, in this order)
    1. crop_black_borders  remove the black frame around the retina
    2. resize              force 512x512 (INTER_AREA keeps small details)
    3. CLAHE               local contrast boost on the L channel of LAB, so small
                           lesions stand out without changing the colors

INPUTS
    data/clean.csv           produced by src/clean_data.py (id_code, diagnosis)
    data/train_images/*.png  the raw images

OUTPUT
    data/processed_512/<id_code>.png   one file per kept image

NOTES
    - Resizing a non-square crop to 512x512 stretches it slightly. This is an
      accepted simplification.
    - Runs are resumable: files that already exist are skipped. Delete
      data/processed_512 to reprocess everything (needed if you change the steps).
    - OpenCV reads and writes BGR, other libraries expect RGB, hence the
      conversions around the processing.

USAGE (from the project root, with the virtual environment active)
    python -m src.preprocess
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def crop_black_borders(img, tol=7):
    """Crop to the bounding box of pixels brighter than tol (RGB image in, RGB out)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol                      # True where there is retina; only used to FIND the box
    if mask.sum() == 0:
        return img                         # fully black image: return unchanged
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def preprocess(img, size=512):
    """Crop, resize to size x size, then apply CLAHE on the L channel (RGB in, RGB out)."""
    img = crop_black_borders(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def main(csv_path=DATA / "clean.csv",
         img_dir=DATA / "train_images",
         out_dir=DATA / "processed_512",
         size=512):
    """Preprocess every image listed in csv_path and save it to out_dir."""
    df = pd.read_csv(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for code in tqdm(df["id_code"]):
        dst = out / f"{code}.png"
        if dst.exists():                   # lets you resume if interrupted
            continue
        img = cv2.imread(str(Path(img_dir) / f"{code}.png"))
        if img is None:
            print(f"Could not read {code}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess(img, size)
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    n_files = len(list(out.glob("*.png")))
    print(f"\nImages in clean.csv: {len(df)} | files in {out.name}: {n_files}")


if __name__ == "__main__":
    main()