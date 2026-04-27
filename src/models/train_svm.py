from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

from src.data.preprocess import build_dataset
from src.config import DATA_DIR, MODEL_PATH

def train():
    X, y = build_dataset(DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    model = svm.SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    train()