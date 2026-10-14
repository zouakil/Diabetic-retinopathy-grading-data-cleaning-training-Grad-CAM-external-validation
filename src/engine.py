"""
engine.py - Training and evaluation loops.

CONTENTS
    train_one_epoch  one pass over the training loader, with mixed precision (AMP).
                     Returns the mean training loss.
    evaluate         one pass over a loader without gradients. Returns a dict:
                     loss, acc, qwk, auc_macro, auc_referable, sensitivity (recall
                     per grade), confusion (5x5 matrix), probs and y_true.

METRICS
    qwk            quadratic weighted kappa on the predicted grade (main metric)
    auc_macro      mean of the five one-vs-rest AUCs
    auc_referable  AUC of "grade 2 or higher" using the summed probability of
                   grades 2 to 4
    sensitivity    recall per grade, from the confusion matrix

NOTES
    - Uses CUDA autocast, so it needs a GPU as written.
    - The best epoch is chosen on validation data, so reported metrics are
      slightly optimistic.
"""
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):  # mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_probs.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = probs.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    return {
        "loss": total_loss / n,
        "acc": float((y_pred == y_true).mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "auc_macro": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
        # "referable DR" = grade 2 or higher: the clinically important yes/no question
        "auc_referable": float(roc_auc_score(y_true >= 2, probs[:, 2:].sum(axis=1))),
        "sensitivity": (cm.diagonal() / cm.sum(axis=1)).round(3),  # recall per grade
        "confusion": cm,
        "probs": probs,
        "y_true": y_true,
    }