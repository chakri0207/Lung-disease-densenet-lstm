# 🫁 Lung Sound Classification

### DenseNet121 + BiLSTM (Mel Spectrogram Based)

------------------------------------------------------------------------

## 📌 Project Overview

This project implements an AI-based Lung Sound Classification system
using:

-   🎵 Mel Spectrogram feature extraction\
-   🧠 DenseNet121 (CNN) for spatial feature learning\
-   ⏳ BiLSTM for temporal modeling\
-   ⚖️ Focal Loss for class imbalance handling\
-   🌐 Gradio Web Interface for live inference

The system classifies lung sounds into 6 respiratory conditions:

-   Asthma\
-   COPD\
-   Heart Failure\
-   Lung Fibrosis\
-   Normal\
-   Pneumonia

> ⚠️ This is an educational student project and not a medical device.

------------------------------------------------------------------------

## 🧠 Architecture

Audio (.wav)\
↓\
Resample + Normalize\
↓\
Split into 5 segments (2s each)\
↓\
Mel Spectrogram (128 mels)\
↓\
Resize to 224×224\
↓\
DenseNet121 (Feature Extractor)\
↓\
BiLSTM (Temporal Learning)\
↓\
Fully Connected Layer\
↓\
Softmax (6 classes)

------------------------------------------------------------------------

## 📂 Project Structure

lung_sound_project/
│
├── app/
│   └── gradio_app.py          # Web UI
│
├── src/
│   ├── features.py            # Audio + Mel feature extraction
│   ├── model.py               # DenseNet + BiLSTM model
│   ├── inference.py           # Model loading + prediction
│   ├── dataset.py             # Training dataset class
│   ├── config.py              # Config definitions
│   └── label_map.py
│
├── data/
│   ├── raw/                   # ICBHI + Fraiwan datasets
│   └── processed/
│       └── manifests/
│
├── models/
│   ├── best_model.pth
│   └── config.json
│
├── notebooks/
│   ├── 01_build_manifest.ipynb
│   ├── 02_train_model.ipynb
│   ├── 04_train_focal_loss.ipynb
│   └── 03_evaluate.ipynb
│
├── requirements.txt
└── README.md

------------------------------------------------------------------------

## 📊 Dataset Used

### 1️⃣ ICBHI 2017 Respiratory Sound Database

-   920 audio recordings\
-   Patient-level diagnosis labels

### 2️⃣ Fraiwan Lung Sound Dataset

-   336 recordings\
-   Includes Heart Failure & Lung Fibrosis

------------------------------------------------------------------------

## ⚙️ Preprocessing Pipeline

-   Resampled to 22,050 Hz\
-   Peak normalization\
-   Center crop / padding\
-   Split into 5 segments\
-   Mel spectrogram (n_mels=128, n_fft=2048, hop_length=512)\
-   Resize to 224×224\
-   ImageNet normalization

------------------------------------------------------------------------

## 🏋️ Training Strategy

### Phase 1

-   Freeze DenseNet backbone\
-   Train BiLSTM + classifier\
-   LR = 1e-3

### Phase 2

-   Unfreeze last DenseNet block\
-   Fine-tune\
-   LR = 1e-5

Class imbalance handled using Focal Loss.

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   Accuracy\
-   Precision\
-   Recall\
-   F1-score (macro)\
-   Confusion Matrix

Current test performance: - Accuracy ≈ 81%\
- Macro F1 ≈ 0.37

------------------------------------------------------------------------

## 🚀 How to Run

### Install dependencies

pip install -r requirements.txt

### Run Web App

python app/gradio_app.py

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is for educational and research purposes only.\
It is not a certified medical diagnostic tool.
