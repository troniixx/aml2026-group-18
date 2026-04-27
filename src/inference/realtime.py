import cv2
import joblib
from src.features.landmarks import extract_landmarks
from src.config import MODEL_PATH

def run():
    model = joblib.load(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_landmarks(frame)

        if features is not None:
            pred = model.predict([features])[0]

            cv2.putText(
                frame, pred, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

        cv2.imshow("Naruto Handsigns", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()