import os
import numpy as np
from src.features.landmarks2 import extract_landmarks
import cv2
from tqdm import tqdm
from src.config import NEW_CLASSES, CLASSES

def build_dataset(data_dir):
    X, y = [], []
    cwd = os.getcwd()

    plotter = []

    classes = CLASSES if 'zero' in os.listdir(data_dir) else CLASSES[:-1]
    classes = CLASSES[:-1]

    for label in classes:
        class_dir = cwd + '/' + data_dir + '/' + label
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
    
    X_clean, y_clean = [], []

    for x, label in zip(X, y):
        if x is not None and len(x) > 0:
            X_clean.append(x)
            y_clean.append(label)

    return np.array(X_clean), np.array(y_clean), plotter

def build_dataset_other(data_dir):
    X, y = [], []
    cwd = os.getcwd()

    for pic in tqdm(os.listdir(data_dir), desc=f"Processing additional data"):
        if pic.endswith('.jpg'):
            img_path = os.path.join(data_dir, pic)
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
    
    return np.array(X_clean), np.array(y_clean)