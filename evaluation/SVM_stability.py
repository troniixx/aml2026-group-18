import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np

from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SVM_ROOT = PROJECT_ROOT / "SVM_model"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SVM_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SVM_ROOT:", SVM_ROOT)

from model_defs import SEAL_CLASSES
from SVM_model.config import CLASSES
from SVM_model.data.preprocess1 import build_dataset as build_dataset1
# 如果你要测试第二种特征，可以打开这一行
# from SVM_model.data.preprocess2 import build_dataset as build_dataset2


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATA_ROOT = Path("/Users/cynthiaj/Desktop/Advanced ML/project/aml2026-group-18/data/final")
TRAIN_DIR_CROPPED = DATA_ROOT / "train_cropped"

N_SPLITS = 5
RANDOM_STATE = 42


def normalize_labels_to_names(y):
    """
    Make y labels comparable.

    If y is numeric: 0,1,2 -> class names
    If y is already string: keep as string
    """
    y = np.array(y)

    if np.issubdtype(y.dtype, np.integer):
        return np.array(SEAL_CLASSES)[y]
    else:
        return y.astype(str)


def prepare_score_matrix(model, X_val):
    """
    Returns y_score with columns aligned to SEAL_CLASSES.
    Removes extra classes such as '' or 'zero' if present.
    """
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)
    else:
        y_score = model.decision_function(X_val)

    model_classes = [str(c) for c in list(model.classes_)]

    print("model_classes before:", model_classes)
    print("y_score shape before:", y_score.shape)

    # Remove empty extra class if present
    if "" in model_classes:
        empty_col = model_classes.index("")
        print(f"Removing extra empty class column at index: {empty_col}")
        y_score = np.delete(y_score, empty_col, axis=1)
        model_classes.pop(empty_col)

    # Remove zero class if present
    if "zero" in model_classes:
        zero_col = model_classes.index("zero")
        print(f"Removing extra zero class column at index: {zero_col}")
        y_score = np.delete(y_score, zero_col, axis=1)
        model_classes.pop(zero_col)

    print("model_classes after:", model_classes)
    print("y_score shape after:", y_score.shape)

    if model_classes != list(SEAL_CLASSES):
        raise ValueError(
            "Score column order mismatch!\n"
            f"model_classes: {model_classes}\n"
            f"SEAL_CLASSES: {SEAL_CLASSES}"
        )

    return y_score


def run_svm_5fold_stability(dataset_builder, model_name="SVM"):
    print(f"\n========== Building dataset for {model_name} ==========")

    result = dataset_builder(data_dir=str(TRAIN_DIR_CROPPED))

    if len(result) == 3:
        X, y, plotter = result
        print("Class counts from preprocess:")
        print(plotter)
    else:
        X, y = result

    X = np.array(X)
    y_names = normalize_labels_to_names(y)

    print("X shape:", X.shape)
    print("y shape:", y_names.shape)
    print("y dtype:", y_names.dtype)

    print("\nLabel distribution:")
    counts = Counter(y_names)
    for cls in SEAL_CLASSES:
        print(cls, counts[cls])

    # Check only 12 seal classes
    valid_mask = np.isin(y_names, SEAL_CLASSES)
    X = X[valid_mask]
    y_names = y_names[valid_mask]

    print("\nAfter filtering to 12 seal classes:")
    print("X shape:", X.shape)
    print("y shape:", y_names.shape)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_top1 = []
    fold_top3 = []
    fold_cms = []

    indices = np.arange(len(y_names))

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_names)):
        print(f"\n========== {model_name} | Fold {fold + 1}/{N_SPLITS} ==========")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_names[train_idx], y_names[val_idx]

        print("Train shape:", X_train.shape)
        print("Val shape:", X_val.shape)

        model = svm.SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val).astype(str)

        top1 = accuracy_score(y_val, y_pred)

        y_score = prepare_score_matrix(model, X_val)

        top3 = top_k_accuracy_score(
            y_val,
            y_score,
            k=3,
            labels=SEAL_CLASSES
        )

        cm = confusion_matrix(
            y_val,
            y_pred,
            labels=SEAL_CLASSES
        )

        fold_top1.append(top1)
        fold_top3.append(top3)
        fold_cms.append(cm)

        print(
            f"{model_name} Fold {fold + 1} result | "
            f"top1 {top1:.4f} | top3 {top3:.4f}"
        )
        print("Confusion matrix:")
        print(cm)

    print(f"\n========== {model_name} Stability Result ==========")
    print("Top-1 scores:", fold_top1)
    print("Top-1 mean:", np.mean(fold_top1))
    print("Top-1 std:", np.std(fold_top1))

    print("Top-3 scores:", fold_top3)
    print("Top-3 mean:", np.mean(fold_top3))
    print("Top-3 std:", np.std(fold_top3))

    mean_cm = np.mean(fold_cms, axis=0)
    print("Mean confusion matrix:")
    print(mean_cm)

    return fold_top1, fold_top3, fold_cms


if __name__ == "__main__":
    svm_top1, svm_top3, svm_cms = run_svm_5fold_stability(
        dataset_builder=build_dataset1,
        model_name="SVM Model 1"
    )

    # svm2_top1, svm2_top3, svm2_cms = run_svm_5fold_stability(
    #     dataset_builder=build_dataset2,
    #     model_name="SVM Model 2"
    # )