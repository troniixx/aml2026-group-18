# Naruto Hand Seal Detector
Group project for the "Advanced Machine Learning (FS26)" course. 
The system detects hand seals (jutsu) from a camera or dataset using a combination of MediaPipe hand landmarks, PyTorch image classifiers (Swin, MobileNetV2), and an optional SVM classifier for lightweight inference.

**Group members:** Jennifer Leleany Meyer, Ishana Rana, Mert Erol, Chenxi Jiang, Aejong Shin

**Repository layout (top-level):**
- `app/` — main application and runtime components
- `Advanced_models/` — PyTorch model definitions and weights
- `SVM_model/` — SVM models and inference helpers
- `datasets/`, `Dataset_Merger/`, `EDA/`, `evaluation/` — data and analysis utilities
- `project_presentation` - Presentation containing further details
---
**Quick Start (Windows)**

1. Create and activate the virtual environment (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app (from repository root):

```powershell
cd app
python main.py
```

**Configuration**
- The app is driven by [app/config.yaml](app/config.yaml). Models are defined there with keys like `model_1`, `model_2`, `model_3` and a `model_type` field (`pytorch` or `svm`). Change thresholds and the active model in this file.
- Example SVM entry in `app/config.yaml`:


**Usage Notes**
- Runtime model switching is supported. The app loads models on demand and switches inference pipelines between PyTorch models (Swin / MobileNetV2) and the SVM branch.
- The `confidence`/`threshold` value in `app/config.yaml` is applied to both PyTorch and SVM outputs. SVM uses `predict_proba` when available, otherwise decision scores are used.

**SVM compatibility**
- SVM models are saved as pickles under `SVM_model/models/`. Ensure scikit-learn compatibility when loading a pickle (a warning is shown if versions differ).

---
## Our Models

| *Model* | **scikit-learn.SVM** | **MobileNetV2** | **Swin Transformer** |
|---|---|---|---|
| *Architecture Type* | Traditional (Kernel-based) | Lightweight CNN | Pure Vision Transformer |
| *Input Strategy* | Raw Landmarks | MediaPipe Crops | Local Attention |
| *Key Advantage* | Memory Efficient | Inference Speed | Interpretability |
| *Accuracy* | 68.23% | 98.40% | 98.00%+ |

