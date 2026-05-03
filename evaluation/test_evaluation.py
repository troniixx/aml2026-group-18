import torch
from torch.utils.data import Dataset, DataLoader
from metrics import evaluate_model
from cross_validation import run_5fold_cv


class FakeSealDataset(Dataset):
    def __init__(self, n=120, num_classes=12):
        self.num_classes = num_classes
        self.targets = [i % num_classes for i in range(n)]
        self.classes = [
            "rat", "ox", "tiger", "rabbit",
            "dragon", "snake", "horse", "goat",
            "monkey", "rooster", "dog", "pig"
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        label = self.targets[idx]
        return image, label


class DummyCNN(torch.nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


def dummy_train_model(model, train_loader, device):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        _ = model(images)
        break


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FakeSealDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = DummyCNN().to(device)

top1, top3, cm = evaluate_model(model, loader, device)

print("Top-1:", top1)
print("Top-3:", top3)
print("Confusion matrix shape:", cm.shape)

fold_top1, fold_top3 = run_5fold_cv(
    dataset=dataset,
    model_class=DummyCNN,
    train_model=dummy_train_model,
    device=device,
    num_classes=12
)

print("Test completed successfully.")

