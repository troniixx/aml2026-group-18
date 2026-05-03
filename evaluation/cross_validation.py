from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
from metrics import evaluate_model

def run_5fold_cv(dataset, model_class, train_model, device, num_classes=12):
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    fold_top1_results = []
    fold_top3_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\nFold {fold + 1}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=32,
            shuffle=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=32,
            shuffle=False
        )

        model = model_class().to(device)

        train_model(model, train_loader, device)

        top1, top3, cm = evaluate_model(
            model,
            val_loader,
            device,
            num_classes=num_classes
        )

        fold_top1_results.append(top1)
        fold_top3_results.append(top3)

        print(f"Fold {fold + 1} Top-1 Accuracy: {top1:.4f}")
        print(f"Fold {fold + 1} Top-3 Accuracy: {top3:.4f}")

    print("\n5-fold CV results")
    print("Top-1 accuracies:", fold_top1_results)
    print("Top-1 mean:", np.mean(fold_top1_results))
    print("Top-1 std:", np.std(fold_top1_results))

    print("Top-3 accuracies:", fold_top3_results)
    print("Top-3 mean:", np.mean(fold_top3_results))
    print("Top-3 std:", np.std(fold_top3_results))

    return fold_top1_results, fold_top3_results

if __name__ == "__main__":
    cnn_fold_top1, cnn_fold_top3 = run_5fold_cv(
        dataset=dataset,
        model_class=CNNModel,
        train_model=train_model,
        device=device,
        num_classes=12
    )