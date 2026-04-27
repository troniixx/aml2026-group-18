import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    landmarks = []

    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks)