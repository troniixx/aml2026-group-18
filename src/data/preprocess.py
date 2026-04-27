import os
import numpy as np
from src.features.landmarks import extract_landmarks
import cv2

def build_dataset(data_dir):
    X, y = [], []

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)

            features = extract_landmarks(image)

            if features is not None:
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)