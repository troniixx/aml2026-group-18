import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = np.array([
    [21, 0, 7, 0, 0, 0, 0, 1, 0, 0, 1, 2],
    [0, 13, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 31, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 4, 37, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 3, 12, 8, 4, 4, 4, 0, 1, 1, 0],
    [0, 0, 2, 3, 0, 30, 0, 1, 0, 0, 0, 8],
    [1, 1, 6, 4, 4, 0, 4, 2, 1, 0, 1, 0],
    [8, 0, 3, 3, 0, 4, 0, 4, 3, 0, 2, 1],
    [3, 0, 2, 0, 0, 2, 0, 0, 20, 0, 0, 21],
    [0, 0, 1, 8, 0, 3, 0, 0, 10, 4, 12, 1],
    [1, 0, 0, 7, 0, 0, 0, 0, 2, 8, 21, 0],
    [0, 0, 6, 1, 0, 0, 0, 0, 3, 0, 0, 23]
])

class_names = [
    "bird", "boar", "dog", "dragon",
    "hare", "horse", "monkey", "ox",
    "ram", "rat", "snake", "tiger"
]

top1 = 0.508235294117647

fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(
    cmap="viridis",
    ax=ax,
    xticks_rotation=45,
    values_format="d",
    colorbar=True
)

ax.set_title(f"SVM Model 1 — Test Accuracy: {top1:.3f}")
plt.tight_layout()
plt.show()