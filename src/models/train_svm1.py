from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.preprocess1 import build_dataset, build_dataset_other
from src.config import DATA_DIR_TRAIN, DATA_DIR_TEST, MODEL_PATH1, CLASSES, NEW_DATA_DIR_TRAIN, NEW_DATA_DIR_VAL, NEW_DATA_DIR_TEST, DATA_DIR_OWN

def train():
    print('Building training dataset...')
    X_train, y_train, plotter_train = build_dataset(data_dir=DATA_DIR_TRAIN)
    print('Building test dataset...')
    X_test, y_test, plotter_test = build_dataset(data_dir=DATA_DIR_TEST)
    print('Building training dataset...')
    X_train_new, y_train_new = build_dataset_other(data_dir=NEW_DATA_DIR_TRAIN)
    print('Building training dataset...')
    X_train_new_val, y_train_new_val = build_dataset_other(data_dir=NEW_DATA_DIR_VAL)
    print('Building training dataset...')
    X_train_new_test, y_train_new_test = build_dataset_other(data_dir=NEW_DATA_DIR_TEST)
    print('Building training dataset...')
    X_train_own, y_train_own, plotter_train_own = build_dataset(data_dir=DATA_DIR_OWN)

    X_1 = np.concatenate((X_train, X_train_new))
    y_1 = np.concatenate((y_train, y_train_new))
    X_2 = np.concatenate((X_1, X_train_new_val))
    y_2 = np.concatenate((y_1, y_train_new_val))
    X_3 = np.concatenate((X_2, X_train_new_test))
    y_3 = np.concatenate((y_2, y_train_new_test))
    X_4 = np.concatenate((X_3, X_test))
    y_4 = np.concatenate((y_3, y_test))
    X_5 = np.concatenate((X_4, X_train_own))
    y_5 = np.concatenate((y_4, y_train_own))

    X_train, X_test, y_train, y_test = train_test_split(X_5, y_5, test_size=0.25, random_state=42)

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

    # plt.show()
    ################################################################

    model = svm.SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc*100:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH1), exist_ok=True)
    joblib.dump(model, MODEL_PATH1)

if __name__ == "__main__":
    train()