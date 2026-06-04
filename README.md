# 🖥️ LandMark: Facial Keypoint Detection
### CNN-based Facial Landmark Detection using TensorFlow & Keras

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=flat-square)

> A CNN-based facial keypoint detection pipeline that identifies 15 facial landmarks (30 coordinates) from grayscale images using the Kaggle Facial Keypoints Detection dataset — featuring two model architectures, data augmentation, training callbacks, and full prediction visualization.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Sample Output](#-sample-output)
- [Author](#-author)

---

## 🔭 Overview

Facial landmark detection is a foundational task in computer vision with applications in face recognition, emotion analysis, augmented reality, and driver monitoring systems. This project builds a complete end-to-end pipeline to detect **15 facial keypoints** (eyes, eyebrows, nose tip, and mouth corners) from 96×96 grayscale images.

Two CNN architectures are implemented and compared:
- A **Basic CNN** with 5 convolutional blocks and fully connected layers
- An **Advanced CNN** with residual (skip) connections and Global Average Pooling

All experiments use a fixed random seed of 42 for reproducibility.

---

## ✨ Features

- 📌 Detects **15 facial keypoints** (30 x,y coordinates) from static images
- 🧠 Two model architectures — **Basic CNN** and **Residual CNN**
- 📊 Full **EDA** — missing value analysis, keypoint coordinate distribution
- 🔁 **Data augmentation** via Keras `ImageDataGenerator`
- ⚙️ Training callbacks — `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- 📈 Training/validation **loss and MAE curves**
- 🟢🔴 Side-by-side **ground truth vs prediction** visualization
- 📐 **Pixel-distance accuracy** metric (% keypoints within N-pixel threshold)
- 💾 Automatic model saving as `.h5`

---

## 📦 Dataset

- **Source:** [Kaggle — Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection)
- **Images:** 96×96 grayscale images stored as pixel strings in CSV
- **Labels:** 15 facial keypoints = 30 coordinates (x,y pairs)
- **Subset used:** 1,000 samples (randomly selected, seed=42)
- **Missing values:** Handled by filling with column mean
- **Normalization:** Images scaled to [0,1], keypoints scaled to [0,1] (divided by 96)

**Keypoints detected:**

| Region | Keypoints |
|---|---|
| Eyes | Left/right eye center, inner & outer corners (6 points) |
| Eyebrows | Left/right inner & outer ends (4 points) |
| Nose | Nose tip (1 point) |
| Mouth | Left/right corners, top & bottom lip center (4 points) |

---

## 🧠 Model Architectures

### Basic CNN

```
Input (96×96×1)
→ Conv2D(32) → BN → MaxPool → Dropout(0.2)
→ Conv2D(64) → BN → MaxPool → Dropout(0.2)
→ Conv2D(128) → BN → MaxPool → Dropout(0.3)
→ Conv2D(256) → BN → MaxPool → Dropout(0.3)
→ Conv2D(512) → BN → MaxPool → Dropout(0.4)
→ Flatten → Dense(1024) → BN → Dropout(0.5)
→ Dense(512) → BN → Dropout(0.5)
→ Dense(30) [linear output]
```

### Advanced Residual CNN

```
Input (96×96×1)
→ Conv2D(32, 7×7) → BN → MaxPool
→ Residual Block (64 filters)  → MaxPool → Dropout(0.25)
→ Residual Block (128 filters) → MaxPool → Dropout(0.25)
→ Residual Block (256 filters) → MaxPool → Dropout(0.30)
→ GlobalAveragePooling2D
→ Dense(512) → BN → Dropout(0.5)
→ Dense(256) → BN → Dropout(0.5)
→ Dense(30) [linear output]
```

**Training configuration:**

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | Mean Squared Error (MSE) |
| Metric | Mean Absolute Error (MAE) |
| Epochs | 80 (with early stopping) |
| Batch size | 32 |
| Early stopping patience | 20 |
| LR reduction patience | 8, factor 0.2 |

---

## 📁 Project Structure

```
landmark-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE.md
│
├── LandMarkDFacePoints.py     # Full pipeline — data loading, models, training, evaluation
│
└── Image/                     # Sample output images with overlaid keypoints
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/prabhu-313/landmark-detection.git
cd landmark-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `training.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data) and place them in the root directory:

```
landmark-detection/
├── training.csv      ← place here
├── test.csv          ← place here
└── LandMarkDFacePoints.py
```

> **Note:** If the dataset files are not found, the script automatically generates synthetic sample data for demonstration so you can still run and explore the full pipeline.

### 4. Run the pipeline

```bash
python LandMarkDFacePoints.py
```

This runs all 12 steps automatically: data loading → preprocessing → EDA → model creation → training → evaluation → visualization → model saving.

### 5. Switch model architecture (optional)

Inside `LandMarkDFacePoints.py`, in the `main()` function, swap the model by commenting/uncommenting:

```python
# Basic CNN (default)
model = create_basic_cnn_model(input_shape, num_keypoints)

# Advanced Residual CNN
# model = create_advanced_cnn_model(input_shape, num_keypoints)
```

---

## 🖼️ Sample Output

Sample processed images with detected keypoints are in the [`Image/`](./Image) folder.

- 🟢 **Green dots** = Ground truth keypoints
- 🔴 **Red crosses** = Model predictions

---

## 👤 Author

**Prabhupada Samantaray**
B.Tech CSE, KIIT University
[GitHub](https://github.com/prabhu-313) · [LinkedIn](https://www.linkedin.com/in/prabhupada-samantaray-13apr2002/) · [Email](mailto:psray313@gmail.com)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE.md).
