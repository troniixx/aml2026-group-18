import cv2
import os

def collect_images(label, save_dir="data/train", num_samples=200):
    os.makedirs(f"{save_dir}/{label}", exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Collecting", frame)

        key = cv2.waitKey(1)
        if key == ord("s"):  # press 's' to save
            path = f"{save_dir}/{label}/{count}.jpg"
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            count += 1

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()