# Diabetic Retinopathy Grading

A PyTorch project for five-class diabetic retinopathy grading from retinal fundus photographs.
Training and internal validation use APTOS 2019; IDRiD is reserved for external validation.
External validation failed the pre-specified keep rule (decision: retrain; not carried out within the project's time budget).

Student project, Digital Health Engineering, UM6SS. Author: Zakaria OUAKIL
License: MIT 

## Key results

### Internal validation: APTOS

| Metric | Result |
|---|---:|
| Images | 701 |
| Accuracy | 0.862 |
| QWK | 0.926 |
| Macro AUC | 0.925 |
| Referable DR AUC | 0.976 |
| Referable sensitivity / specificity at 0.5 | 0.920 / 0.965 |

Referable DR = grade 2 or higher, predicted when the summed probability of grades 2 to 4 is at least 0.5. The best epoch was selected on this validation set, so the internal figures are slightly optimistic.

### External validation: IDRiD

| Metric | Result (95% bootstrap CI where supplied) |
|---|---:|
| Images | 516 |
| Accuracy | 0.405 (0.362–0.450) |
| QWK | 0.650 (0.590–0.702) |
| Referable DR AUC | 0.910 (0.883–0.934) |
| Sensitivity / specificity at 0.5 | 0.678 (0.628–0.730) / 0.969 (0.941–0.990) |
| Macro AUC | 0.791 |

Referable DR = grade 2 or higher, predicted when the summed probability of grades 2 to 4 is at least 0.5.

## Data quality findings

APTOS cleaning found 123 exact duplicate groups (251 images). Thirty groups had contradictory labels; all 62 images in those groups were removed. One image was retained from each other group, removing 96 extra copies. The cleaned set contains 3,504 images. No unreadable, too-small, or empty images were found. A further 114 dark, bright, blurry, or low-retina-area images were reviewed visually and retained.

The original random split placed 41 duplicate groups across training and validation, creating leakage. The first project version used that split; its results are preliminary and should not be compared with the final results. The archived model artifacts in archive_v1/ are kept locally and excluded from Git; a directory listing confirmed that archive_v1/ contains only the models folder and no archived notebooks. The final stratified split has zero overlap by ID and image hash.

## Method

Images are border-cropped, resized to 512×512, and processed with CLAHE on the L channel in LAB color space. Non-square crops are slightly stretched. The cleaned APTOS data uses a stratified 80/20 split with seed 42: 2,803 training and 701 validation images.

The classifier is a 5-class timm EfficientNet-B0 pretrained on ImageNet, with 4.01 million parameters. Grades are 0 No DR, 1 Mild, 2 Moderate, 3 Severe, and 4 Proliferative. Referable DR means grade 2 or higher. Training used an inverse-square-root weighted sampler, label smoothing 0.1, AdamW (learning rate 3e-4, weight decay 1e-2), cosine schedule, batch size 16, and mixed precision. Training ran up to 15 epochs with patience 5 and stopped after 14; epoch 9 was selected by validation QWK. Training-only augmentation used small shifts and zooms, flips, rotations up to 20 degrees, and brightness, contrast, and saturation jitter.

The primary metric is quadratic weighted kappa. Grad-CAM supports qualitative inspection, with a separate masking test to examine sensitivity to pixels outside the retina.

## Explainability findings

Grad-CAM uses the last feature map, which is 16×16 and coarse. Some highlighted regions coincide with possible lesions, while heat also appears on borders and corners. The black background accounts for 13.9% of heat mass and 16.2% of image area (ratio 0.86); the hottest pixel was on the background in 4% of validation images, and more than 25% of heat fell there in 11.6%.

Setting pixels outside the retina to black without retraining changed QWK from 0.925 to 0.908 and referable sensitivity from 0.920 to 0.869 (15 newly missed versus 1 newly caught; exact test p = 0.001). Masking only dark pixels inside the retina had no significant effect (p = 0.125). This is consistent with sensitivity to background pixel values and suggests fragility; it does not prove diagnostic use of background. Two referable cases were missed at every threshold and predicted Mild with probability 0.96–0.97; neither received clinician review.

## External validation and the pre-specified decision rule

IDRiD disease-grading images were not used for training, epoch selection, or threshold selection. Before viewing results, the rule was: KEEP if referable AUC ≥ 0.95, sensitivity at 0.5 ≥ 0.85, and QWK ≥ 0.80; RETRAIN if any is below 0.90, 0.75, and 0.70 respectively; otherwise inconclusive.

The result was RETRAIN. IDRiD QWK was 0.650, referable AUC 0.910, and sensitivity at 0.5 was 0.678. The model was not retrained within the project time budget. Since the retrain decision was informed by IDRiD, any later version will not be a pure zero-shot test. The performance drop is three to four times larger than the train-to-validation gap, which suggests dataset shift (camera, clinic, grading conventions, or image selection) on top of some memorization; the cause was not identified.

## Limitations

- Training used a single dataset.
- Rare grades have few validation examples (35 Severe and 54 Proliferative in APTOS; 25 Mild in IDRiD).
- The best epoch and explored thresholds were selected using validation data; the threshold 0.2 sensitivity result is optimistic.
- Non-square crops are stretched to 512×512.
- Only exact duplicates were detected; contradictory duplicate labels indicate label noise.
- No clinician reviewed hard cases.
- Grad-CAM is coarse and does not establish model reasoning.
- External validation failed the pre-specified keep rule.
- IDRiD comes from a single clinic and camera, with images selected as gradable and of good quality, and its mix (62.6% referable) differs from APTOS validation (39%), so QWK and accuracy are not strictly comparable.
- This is not a clinical tool.

## Repository structure

    dr-project/
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── models/
    │   └── history_v2.csv
    ├── notebooks/
    │   ├── 00_exploration.ipynb
    │   ├── 01_data_cleaning.ipynb
    │   ├── 02_preprocessing.ipynb
    │   ├── 03_training.ipynb
    │   ├── 04_gradcam.ipynb
    │   └── 05_external_validation.ipynb
    └── src/
        ├── __init__.py
        ├── clean_data.py
        ├── dataset.py
        ├── engine.py
        ├── gradcam.py
        ├── make_split.py
        ├── model.py
        ├── prepare_external.py
        └── preprocess.py

data/ and archive_v1/ directories are present locally; their contents are not listed. Data and model weights are not intended for distribution with the repository.

## How to reproduce

Use Python 3.13. Create and activate a virtual environment, then install a CUDA-enabled PyTorch and torchvision build appropriate for the system from pytorch.org before installing the pinned packages in requirements.txt:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    # Install torch and torchvision (CUDA build) from https://pytorch.org/get-started/locally/
    pip install -r requirements.txt
    python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda)"

Download APTOS 2019 Blindness Detection from Kaggle into data/ with train.csv and train_images/. From the project root, run:

    python -m src.clean_data
    python -m src.preprocess
    python -m src.make_split

Run notebooks 00_exploration through 04_gradcam in order. For external validation, download IDRiD “B. Disease Grading” from IEEE DataPort into data/external/idrid/raw/, then run:

    python -m src.prepare_external

Then run notebook 05_external_validation. Always run scripts as python -m src.<name> from the project root. Data and model weights are not in the repository. Notebook 03 trains the model and saves the weights as `models/efficientnet_b0_v2_best.pth`; notebooks 04 and 05 load them.

Sampling and augmentation are random, so retraining will give slightly different numbers than those reported. The reported results come from the model saved by notebook 03 (best epoch 9).

## Next steps

- Remove the background during preprocessing and retrain.
- Use stronger color, contrast, and blur augmentation and less camera-sensitive color normalization.
- Evaluate ordinal regression.
- Adapt using the IDRiD training part, label the result as adapted, then test once on its 103 test images.
- Evaluate a second external dataset, such as Messidor-2.
- Ask a clinician to review hard cases.

## Data sources, citations and license notes

- APTOS 2019: [Kaggle competition](https://www.kaggle.com/c/aptos2019-blindness-detection). Follow its terms; data is not redistributed here.
- IDRiD: Porwal P. et al., “Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research”, Data 2018, 3(3), 25 ([DOI](https://doi.org/10.3390/data3030025)). IDRiD is CC BY 4.0; credit is required.
- Tan and Le (2019), EfficientNet.
- Selvaraju et al. (2017), Grad-CAM.
- Ross Wightman, PyTorch Image Models (timm).

Project code license: MIT (see `LICENSE`). The datasets and dependencies have their own terms, listed above.

Recorded Windows environment: NVIDIA RTX 3070 Laptop GPU, Python 3.13, torch 2.14.1+cu126, torchvision 0.29.1+cu126, CUDA 12.6. The version values are from the existing environment metadata.

## Disclaimer

This is a research prototype, not a medical device, and is not for clinical use.
