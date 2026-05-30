import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SVM_ROOT = PROJECT_ROOT / "SVM_model"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SVM_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SVM_ROOT:", SVM_ROOT)

from model_defs import SEAL_CLASSES

from SVM_model.data.preprocess1 import build_dataset as build_dataset1
from SVM_model.data.preprocess2 import build_dataset as build_dataset2
from SVM_model.config import MODEL_PATH1, MODEL_PATH2, CLASSES

DATA_DIR = "/Users/cynthiaj/Desktop/Advanced ML/project/aml2026-group-18/data/final/test_cropped"


def evaluate_svm(model_path, model_name="SVM"):
    print(f"\n===== Evaluating {model_name} =====")

    model_path = Path(model_path)
    print("Loading SVM model from:", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found: {model_path}")

    model = joblib.load(model_path)
    print("Model expects n_features:", model.n_features_in_)

    print("Building test dataset...")
    if model_name == "SVM Model 1":
        result = build_dataset1(data_dir=DATA_DIR)
    elif model_name == "SVM Model 2":
        result = build_dataset2(data_dir=DATA_DIR)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if len(result) == 3:
        X_test, y_test, _ = result
    else:
        X_test, y_test = result

    X_test = np.array(X_test)
    print("Current X_test features:", X_test.shape[1])
    y_test = np.array(y_test)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("y_test dtype:", y_test.dtype)
    print("first 10 y_test:", y_test[:10])

    # SVM config has an extra zero class, but evaluation uses 12 classes
    SVM_EVAL_CLASSES = [c for c in CLASSES if c != "zero"]

    print("SVM classes from config:", CLASSES)
    print("SVM eval classes:", SVM_EVAL_CLASSES)
    print("Expected classes:", SEAL_CLASSES)

    if list(SVM_EVAL_CLASSES) != list(SEAL_CLASSES):
        raise ValueError(
            "Class order mismatch!\n"
            f"SVM_EVAL_CLASSES: {SVM_EVAL_CLASSES}\n"
            f"SEAL_CLASSES: {SEAL_CLASSES}"
        )

    print("Model classes_:", model.classes_)

    # Convert y_test to class names if needed
    if np.issubdtype(y_test.dtype, np.integer):
        y_true_names = np.array(SEAL_CLASSES)[y_test]
    else:
        y_true_names = y_test.astype(str)

    # SVM prediction gives class names directly
    y_pred_names = model.predict(X_test).astype(str)

    print("Unique y_true:", np.unique(y_true_names))
    print("Unique y_pred:", np.unique(y_pred_names))

    # Top-1 accuracy
    top1 = accuracy_score(y_true_names, y_pred_names)

    # Top-3 scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)

    print("y_score shape before:", y_score.shape)

    model_classes = list(model.classes_)
    model_classes = [str(c) for c in model_classes]
    print("model_classes before:", model_classes)

    # model has an extra empty class: ''
    if "" in model_classes:
        empty_col = model_classes.index("")
        print(f"Removing extra empty class column at index: {empty_col}")
        y_score = np.delete(y_score, empty_col, axis=1)
        model_classes.pop(empty_col)

    # If there is still a zero class, remove it
    if "zero" in model_classes:
        zero_col = model_classes.index("zero")
        print(f"Removing extra zero class column at index: {zero_col}")
        y_score = np.delete(y_score, zero_col, axis=1)
        model_classes.pop(zero_col)

    print("model_classes after:", model_classes)
    print("y_score shape after:", y_score.shape)

    # Check that probability columns match SEAL_CLASSES
    if model_classes != list(SEAL_CLASSES):
        raise ValueError(
            "Score column order mismatch!\n"
            f"model_classes: {model_classes}\n"
            f"SEAL_CLASSES: {SEAL_CLASSES}"
        )

    top3 = top_k_accuracy_score(
        y_true_names,
        y_score,
        k=3,
        labels=SEAL_CLASSES
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_true_names,
        y_pred_names,
        labels=SEAL_CLASSES
    )

    print(f"{model_name} Top-1:", top1)
    print(f"{model_name} Top-3:", top3)
    print(f"{model_name} Confusion matrix shape:", cm.shape)
    print(cm)

    empty_pred_count = np.sum(y_pred_names == "")
    print("Number of predictions as empty extra class:", empty_pred_count)

    zero_pred_count = np.sum(y_pred_names == "zero")
    print("Number of predictions as 'zero':", zero_pred_count)


if __name__ == "__main__":
    SVM_MODEL_PATH1 = SVM_ROOT / MODEL_PATH1
    SVM_MODEL_PATH2 = SVM_ROOT / MODEL_PATH2

    print("MODEL_PATH1 from config:", MODEL_PATH1)
    print("Resolved SVM model path 1:", SVM_MODEL_PATH1)

    print("MODEL_PATH2 from config:", MODEL_PATH2)
    print("Resolved SVM model path 2:", SVM_MODEL_PATH2)

    evaluate_svm(SVM_MODEL_PATH1, model_name="SVM Model 1")
    #evaluate_svm(SVM_MODEL_PATH2, model_name="SVM Model 2")