# Group 18

## Naruto Handsign Detection 

### Problemsetting
Given a video $v = \langle f_1, f_2, \ldots, f_N \rangle$, where $f_n$ represents the $n$-th RGB frame of a person performing a hand seal, our goal is to classify the performed hand seal in real time into a label $y \in \mathcal{Y}$, where $\mathcal{Y} = \{y_1, \ldots, y_{12}\} \cup \mathcal{C}$ denotes the set of the 12 basic hand seals and a predefined set $\mathcal{C}$ of named special combinations thereof.
The predicted label $\hat{y}$ is then used to trigger a corresponding visual effect overlaid on the live video feed.

##### Background: Hand Seals in Naruto
In the anime *Naruto Shippuden*, hand seals are specific finger and hand gestures used to shape chakra (an internal energy source) and execute jutsu (ninja techniques). The 12 basic seals are derived from the Chinese zodiac signs and serve as the fundamental building blocks from which arbitrarily complex jutsu sequences can be composed.
This project focuses exclusively on the 12 basic seals and a fixed set of named multi-seal combinations.


### Approach

1. Curate Dataset:
   1. Extend an existing hand gesture dataset with self-recorded samples covering all 12 basic seals to improve diversity and balance.
   2. label data with the software CVAT
    - OR use existing one (like: https://www.kaggle.com/vikranthkanumuru/naruto-hand-sign-dataset)
2. Create and Train Model
   1. Train and evaluate models on classifying a single hand seal from a short video clip (or single frame).
   2. Extend to recognizing ordered sequences
   of seals, mapping known multi-seal combinations to a named jutsu label.
3. Evaluate:
   1.  Quantitative metrics on a held-out test set, a confusion
   matrix analysis, and k-fold cross-validation to account for the limited
   dataset size.

#### Architecture

##### Baseline Model
- **MediaPipe Hands + SVM**
  - Extract 21 hand landmark coordinates $(x, y, z)$
  per frame using MediaPipe Hands, yielding a 63-dimensional feature vector. Train a multi-class SVM on these keypoints. This requires no learned visual features => minimal-complexity reference point.
  
  OR:
- **Single-Frame MobileNetV2**: 
  - A pretrained MobileNetV2 CNN fine-tuned on
  individual frames. Clip-level predictions are obtained by majority vote over frames. 


##### Advanced Model Options

- **CNN + Transformer Hybrid** (primary Stage 1 model): Per-frame visual features are extracted with a frozen or fine-tuned CNN backbone (ResNet-18 or MobileNetV2), producing a sequence of embeddings $\mathbf{h} = \langle \mathbf{h}_1, \ldots, \mathbf{h}_N \rangle \in \mathbb{R}^{N \times d}$. A Transformer encoder with learned positional encodings attends over this sequence to capture temporal dependencies across frames. A linear classification head maps the `[CLS]` token output to $\hat{y} \in \mathcal{Y}$. For Stage 2 sequence recognition, the ordered per-seal predictions $\langle \hat{y}_1, \ldots, \hat{y}_K \rangle$ from Stage 1 are embedded and passed through a second lightweight Transformer (or LSTM) that maps the full seal sequence to a named jutsu label $c \in \mathcal{C}$.

- **YOLO-based Pipeline**: Use a pretrained YOLO model (Ultralytics YOLOv8)
  in two stages: first, detect and localize the hand region in each frame,
  yielding a tight crop $f_n^{\text{crop}} \in \mathbb{R}^{H' \times W' \times 3}$; second, pass the crop through a fine-tuned classifier head for seal prediction $\hat{y}$. This pipeline is particularly suited for the live feed application since YOLO is optimized for real-time inference. If detection and classification quality is sufficient, this may serve as an alternative primary model.



##### Tech Stack

| Component | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3.11 | Primary development language |
| Deep Learning | PyTorch 2.x | Model definition, training loop, autograd |
| High-level Training | FastAI | Rapid prototyping, learning rate finder, training callbacks |
| Pretrained Backbones | `torchvision` | MobileNetV2, ResNet-18 pretrained on ImageNet |
| Object Detection | Ultralytics YOLOv8 | Hand localization and YOLO-based classification pipeline |
| Hand Landmarks | MediaPipe Hands | 21-keypoint extraction for the SVM baseline (63-dim feature vector) |
| Classical ML | scikit-learn | SVM, k-fold CV, accuracy/confusion matrix computation |
| Video & Camera | OpenCV (`cv2`) | Frame capture, preprocessing, live feed overlay of visual effects |
| Data Handling | NumPy, pandas | Array operations, dataset bookkeeping |
| Visualisation | Matplotlib, seaborn | Confusion matrices, training curves |
| Experiment Tracking | Weights & Biases (wandb) | Hyperparameter sweeps, metric logging, run comparison |


##### Metrics
- **Top-1 Accuracy** (primary metric): Since the dataset is balanced by
  construction - accuracy is an unbiased summary metric and the natural choice for this 12-class classification problem.
- **Top-3 Accuracy**: Useful during error analysis to assess whether the
  correct class is at least ranked highly by the model, given the visual
  similarity between certain seals.
- **Confusion Matrix**: To identify which seals are systematically confused
  with one another (e.g. visually similar seals from the zodiac set).
- **k-fold Cross-Validation** ($k = 5$): Applied during model selection and
  hyperparameter tuning to produce stable performance estimates given the
  limited dataset size.






-----------------
## Meme sentiment/toxicity classification

### Problemsetting
Given a meme image $i$, we extract text and visual features from the image. Our goal is that the system should asses and classify the sentiment of said image and predict a label y = {harmful, harmless}, where harmful $\in$ (racism, sexism, mobbing, slurs, violence)

##### Background: harmful memes in social media
Memes are widely shared in social media with a rapid speed to spread across online platforms. While many memes are harmless or humorous, some convey harmful content, including racism, sexism, bullying, slurs, and violence. A efficient harmful meme detection could help limit the spread of such content and support to create a more safer and healthier online environment.

### Approach
1. Preprocessing
   1. download existing dataset found on Kaggle
   2. OCR text extraction from meme images
   3. text cleaning and normalization
   4. image resizing and normalization
2. Modeling
   1. text-only model and image-only model
   2. advanced multimodal (using OCR, sentiment Analysis, Text Extraction, TBD: Facial recognition in images)
3. Evaluation: Metrics for classification problems, manual eval, k-fold cross validation

#### Datasets
- Facebook Hateful Memes Challenge

### Architecture

#### Baseline Model
- Text only: fine-tuned BERT on OCR extracted text from meme images
- Image only: fine-tunes ResNET/ViT on meme images 

#### Advanced Models
- CLIP-based classifier: we apply OCR to extract text from meme images and then a pretrained CLIP model is used to encode both textual and visual features from the meme images. CLIP image encoder and text encoder have aligned embeddings and lie in a shared semantic space. The multimodel representation is constructed by concating text and image embeddings. Then a light classification head (like MLP or linear classifier) is applied.
- VisualBERT: we apply OCR to extract text from meme images and visual features are extracted by a pretrained CNN. The visual and textual embeddings are concatenated into a single sequence. This sequence is passed through a multimodal Transformer encoder, where self-attention layers model interactions across both modalities. The final hidden state corresponding to the [CLS] token is used for classification. Then a light classification head (like MLP or linear classifier) is applied.

#### Metrics
- AUROC, F1
- PR-AUC (for imbalanced classification)
- confusion matrix
- k-fold cross validation
