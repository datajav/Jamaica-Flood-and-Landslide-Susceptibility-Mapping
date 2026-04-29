# 🎭 Facial Emotion Detector

> Real-time facial expression recognition using a fine-tuned MobileNetV2 CNN — built with Python, TensorFlow, and OpenCV.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Overview

This project detects and classifies human facial expressions in real time using a webcam feed. A MobileNetV2 model pretrained on ImageNet is fine-tuned on the **FER2013** dataset to recognise **7 emotions**: angry, disgust, fear, happy, neutral, sad, and surprise.

The pipeline covers everything from raw data preprocessing and class imbalance handling, through two-stage fine-tuning, Grad-CAM explainability, and a live Streamlit demo app.

---

## Demo

<!-- Replace with your own screen recording or GIF -->
![Demo GIF](assets/demo.gif)

> *Real-time inference at ~20 fps on CPU. Bounding box + dominant emotion label + probability bars displayed per face.*

---

## Emotions Detected

| Emotion | Label | Training Samples |
|---------|-------|-----------------|
| 😠 Angry | `angry` | 3,995 |
| 🤢 Disgust | `disgust` | 436 |
| 😨 Fear | `fearful` | 4,097 |
| 😄 Happy | `happy` | 7,215 |
| 😐 Neutral | `neutral` | 4,965 |
| 😢 Sad | `sad` | 4,830 |
| 😲 Surprise | `surprised` | 3,171 |

> ⚠️ Note: Significant class imbalance (especially `disgust`) is addressed using class weights during training.

---

## Project Structure

```
emotion-detector/
├── data/
│   └── fer2013.csv              # Raw dataset (download from Kaggle)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_training.ipynb        # Model training & evaluation
│   └── 03_gradcam.ipynb         # Grad-CAM visualisations
├── src/
│   ├── preprocess.py            # Data loading, augmentation, class weights
│   ├── train.py                 # Two-stage fine-tuning pipeline
│   ├── evaluate.py              # Confusion matrix, classification report
│   └── inference.py             # Single-image prediction utility
├── models/
│   └── emotion_model.h5         # Saved model weights (after training)
├── requirements.txt
└── README.md
```

---

## Model Architecture

Base model: **MobileNetV2** (ImageNet weights, `include_top=False`)

```
MobileNetV2 (frozen → partially unfrozen)
    └── GlobalAveragePooling2D
        └── Dense(256, relu)
            └── Dropout(0.4)
                └── Dense(7, softmax)
```

### Training Strategy

**Stage 1 — Head warm-up** (freeze entire base):
- Epochs: 20 | LR: `1e-3` | Optimizer: Adam

**Stage 2 — Fine-tuning** (unfreeze top ~30 layers):
- Epochs: 30 | LR: `1e-5` | Optimizer: Adam
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

Input images are upscaled from 48×48 to **96×96** and converted to 3-channel RGB for compatibility with MobileNetV2.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~65% |
| Macro F1-Score | ~0.61 |
| Inference Speed | ~20 fps (CPU) |

> Human-level accuracy on FER2013 is estimated at ~65%, making this a competitive result on a notoriously noisy dataset.

### Confusion Matrix

<!-- Replace with your actual image -->
![Confusion Matrix](assets/confusion_matrix.png)

### Grad-CAM Visualisation

Grad-CAM highlights the facial regions most influential in each prediction — helping explain *what* the model is responding to.

<!-- Replace with your actual image -->
![Grad-CAM](assets/gradcam.png)

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/emotion-detector.git
cd emotion-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download [FER2013 from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place `fer2013.csv` in the `data/` folder.

### 4. Train the model

```bash
python src/train.py
```

Or step through the training notebook at `notebooks/02_training.ipynb`.

### 5. Run the webcam app

```bash
streamlit run app.py
```

---

## Requirements

```
tensorflow>=2.10
opencv-python>=4.7
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
streamlit>=1.22
```

---

## Key Design Decisions

**Why MobileNetV2?** Fast enough for real-time CPU inference (~20 fps), well-documented transfer learning behaviour, and small enough to deploy on Hugging Face Spaces without a GPU.

**Why two-stage training?** Immediately fine-tuning the full network risks destroying pretrained weights before the new classification head has stabilised. Training the head first for 20 epochs, then unfreezing and continuing at a much lower learning rate, consistently produces better convergence.

**Why class weights over oversampling?** FER2013's `disgust` class has fewer than 500 samples. Oversampling it heavily risks the model memorising the minority class. Class weights penalise misclassifying underrepresented emotions without duplicating data.

---

## Limitations

- FER2013 contains significant label noise — some images are ambiguous even to humans
- Performance drops on faces that are partially occluded, heavily lit, or not forward-facing
- Haar Cascade face detection can miss faces at sharp angles; MTCNN is a drop-in upgrade for better coverage

---

## Roadmap

- [ ] Swap Haar Cascade for MTCNN face detection
- [ ] Add multi-face support
- [ ] Export to TFLite for mobile deployment
- [ ] Train on AffectNet for better real-world generalisation

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — Kaggle
- [face-api.js](https://github.com/justadudewhohacks/face-api.js) for real-time detection inspiration
- [Keras Applications](https://keras.io/api/applications/mobilenet/) — MobileNetV2 pretrained weights