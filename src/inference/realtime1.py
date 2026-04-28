from email.mime import text

import cv2
import joblib
from src.features.landmarks1 import extract_landmarks
from src.config import MODEL_PATH1

def add_text(frame, text, color=(255, 255, 255), font_scale=1, thickness=2):
    h, w = frame.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )
    
    x = (w - text_width) // 2
    y = h - 20

    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale, color, thickness, 
        cv2.LINE_AA,
    )

def run():
    color_map = {
        'bird': (255, 0, 0), 
        'boar': (0, 255, 0), 
        'dog': (0, 0, 255), 
        'dragon': (255, 255, 0), 
        'ox': (255, 0, 255), 
        'tiger': (0, 255, 255), 
        'snake': (255, 128, 0), 
        'rat': (128, 0, 255), 
        'horse': (0, 128, 255), 
        'monkey': (128, 255, 0), 
        'hare': (255, 0, 128), 
        'ram': (0, 255, 128)
    }

    model = joblib.load(MODEL_PATH1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit()


    while True:
        ret, frame = cap.read() # 20px above bottom
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        features = extract_landmarks(frame)

        if features is not None:
            response = model.predict([features])
            pred = response[0]

            if pred != 'zero':
                add_text(frame, pred, color_map[pred])

        cv2.imshow("Naruto Handsigns", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()