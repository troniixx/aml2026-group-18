from email.mime import text

import cv2
import joblib
from src.features.landmarks1 import extract_landmarks
from src.config import MODEL_PATH1

def run():
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
            pred = model.predict([features])[0]

            if pred != 'zero':
                h, w = frame.shape[:2]
                (text_width, text_height), baseline = cv2.getTextSize(
                    pred,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    2
                )
                
                x = (w - text_width) // 2
                y = h - 20 

                cv2.putText(
                    frame, pred, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1, (255, 255, 255), 2, 
                    cv2.LINE_AA,
                )

        cv2.imshow("Naruto Handsigns", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()