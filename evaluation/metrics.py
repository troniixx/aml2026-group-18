import torch
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(model, dataloader, device, num_classes=12):
    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)          # logits: (batch_size, num_classes)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_scores.append(outputs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores, axis=0)

    top1_acc = accuracy_score(y_true, y_pred)

    top3_acc = top_k_accuracy_score(
        y_true,
        y_score,
        k=3,
        labels=np.arange(num_classes)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    return top1_acc, top3_acc, cm

if __name__ == "__main__":
    class_names = dataset.classes  # have to match 'dataset.class_to_idx'

    cnn_top1, cnn_top3, cnn_cm = evaluate_model(
        cnn_model,
        test_loader,
        device,
        num_classes=12
    )

    print("CNN Test Top-1 Accuracy:", cnn_top1)
    print("CNN Test Top-3 Accuracy:", cnn_top3)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cnn_cm,
        display_labels=class_names
    )

    disp.plot(xticks_rotation=45)
    plt.title("CNN Test Confusion Matrix")
    plt.show()