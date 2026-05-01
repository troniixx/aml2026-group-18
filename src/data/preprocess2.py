import os
import numpy as np
from src.features.landmarks2 import extract_landmarks
import cv2
from tqdm import tqdm
from src.config import NEW_CLASSES, NEW_DATA_DIR_TEST, NEW_DATA_DIR_TRAIN, NEW_DATA_DIR_VAL

def build_dataset(data_dir):
    X, y = [], []
    cwd = os.getcwd()

    plotter = []

    for label in os.listdir(data_dir):
        class_dir = cwd + '\\' + data_dir + '\\' + label
        counter = 0

        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing '{label}'"):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)

            features = extract_landmarks(image)

            if features is not None:
                counter += 1
                X.append(features)
                y.append(label)

        print(f"processed {counter} images for class '{label}'")
        plotter.append([label, counter])

    for d in [NEW_DATA_DIR_TRAIN, NEW_DATA_DIR_TEST, NEW_DATA_DIR_VAL]:
        for pic in tqdm(os.listdir(d), desc=f"Processing additional data"):
            if pic.endswith('.jpg'):
                img_path = os.path.join(d, pic)
                image = cv2.imread(img_path)

            features = extract_landmarks(image)

            if features is not None:
                X.append(features)
                y.append('snake' if pic.split('_')[0] == 'serpent' else pic.split('_')[0])
    
    X_clean, y_clean = [], []

    for x, label in zip(X, y):
        if x is not None and len(x) > 0:
            X_clean.append(x)
            y_clean.append(label)

    return np.array(X_clean), np.array(y_clean), plotter