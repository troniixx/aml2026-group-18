from pyexpat import features

import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # up to 2 hands
    detected_hands = results.multi_hand_landmarks[:2]
    landmarks = []

    for hand in detected_hands:
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

    while len(landmarks) < 2 * 21 * 3:
        landmarks.extend([0.0, 0.0, 0.0])

    # Safety: enforce exact size
    landmarks = landmarks[:2 * 21 * 3]

    return np.array(landmarks)