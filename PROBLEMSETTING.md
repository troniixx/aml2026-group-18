# Group 18

## Naruto Handsign

### Problemsetting
Given a sequence i = < $i_1$, $i_2$, $i_3$, ..., $i_n$ >, where $i_n$ represents the $n_(th)$ frame of a video i of a person doing specific handsigns/gestures from the anime Naruto, our goal is that the system should correctly classify the meaning $y$ of the sequence $i$ and execute a visual effect on a live feed.

### Approach
1. Extend training dataset found online with out own data
2. Build and Training model on recognizing a single gesture
3. (Extending) Model to recognize and combine a sequence of gestures
4. Evaluation suite: Metrics for classification problems, manual eval, k-fold cross validation

### Architecture

#### Baseline Model
- Single-Frame CNN (MobileNet)

#### Advanced Model
- ViT
- CNN + Transformer hybrid
- MediaPipe Hands + SVM

#### Metrics
- Top-1 and Top-5 accuracy, confusion matric
## Meme sentiment/toxicity classification

### Problemsetting
Given a meme image $i$, our goal is that the system should asses and classify the sentiment of said image and predict a label y = {harmful, harmless}, where harmful $\in$ (racism, sexism, mobbing, slurs, violence)

### Approach
1. Take existing dataset found on Kaggle
2. Build and train model (using OCR, sentiment Analysis, Text Extraction, TBD: Facial recognition in images)
3. Evaluation: Metrics for classification problems, manual eval, k-fold cross validation

#### Datasets
- Facebook Hateful Memes Challenge

### Architecture

#### Baseline Model
- Text only: fine-tuned BERT on OCR extracted text
- Image only: fine-tunes ResNET/ViT

#### Advanced Models
- CLIP
- VisualBERT

#### Metrics
- AUROC, F1
