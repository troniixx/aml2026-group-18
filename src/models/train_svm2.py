from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.preprocess2 import build_dataset
from src.config import DATA_DIR_TRAIN, DATA_DIR_TEST, MODEL_PATH2, CLASSES

def train():
    print('Building training dataset...')
    X_train, y_train, plotter_train = build_dataset(data_dir=DATA_DIR_TRAIN)
    print('Building test dataset...')
    X_test, y_test, plotter_test = build_dataset(data_dir=DATA_DIR_TEST)

    # PLOT
    ################################################################
    lab = [l for l, v in plotter_train]
    val1 = [v for l, v in plotter_train]
    val2 = [v for l, v in plotter_test]

    x = np.arange(len(lab))  # positions
    width = 0.35                # bar width

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, val1, width, label='train')
    bars2 = ax.bar(x + width/2, val2, width, label='test')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height}', ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height}', ha='center', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.legend()

    plt.show()
    ################################################################

    model = svm.SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH2), exist_ok=True)
    joblib.dump(model, MODEL_PATH2)

if __name__ == "__main__":
    train()