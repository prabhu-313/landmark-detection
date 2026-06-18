# 🎯 LandMark — Real-Time Facial Landmark Detection

### 68-Point Facial Geometry Mapping · OpenCV + Dlib · EAR Drowsiness Detection · MediaPipe Benchmark

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Dlib](https://img.shields.io/badge/Dlib-ERT%20Shape%20Predictor-22C55E?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=flat-square)

> A complete facial landmark detection pipeline that maps **68 anatomically defined key points** across jaw, eyebrows, eyes, nose, and mouth — with a live webcam deployment app, EAR-based drowsiness detection, video file processing, and a GPU inference benchmark comparing Dlib ERT against MediaPipe Face Mesh.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Results & Benchmarks](#-results--benchmarks)
- [The 68 Landmark Map](#️-the-68-landmark-map)
- [Extensions](#-extensions)
- [Project Structure](#-project-structure)
- [Setup & Usage](#️-setup--usage)
- [Local Deployment](#-local-deployment)
- [Author](#-author)

---

## 🔭 Overview

Facial landmarks are the foundational primitive for virtually every advanced face-aware application — AR face filters (Snapchat, Instagram), driver drowsiness detection, emotion recognition, gaze tracking, and biometric liveness verification.

This project implements the **full detection pipeline**:

1. **Face Detection** — Dlib's HOG + Linear SVM frontal face detector
2. **Landmark Localization** — Dlib's pre-trained Ensemble of Regression Trees (ERT) shape predictor, 68 points per face, trained on the iBUG 300-W dataset
3. **Visualization** — OpenCV drawing primitives with bounding box + landmark overlays
4. **EAR Drowsiness Extension** — Eye Aspect Ratio computation for real-time eye closure monitoring
5. **Benchmark** — GPU-accelerated image resizing (PyTorch) + Dlib inference timing across 5 resolutions

All three modes are demonstrated: **static image**, **live Colab webcam capture**, and **video file processing**.

---

## 🏗 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                            │
│   Static Image  │  Colab Webcam (JS Bridge)  │  Video File     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Resize (500px │
                    │  width, aspect │
                    │  ratio kept)   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │ BGR → Grayscale│
                    │ (np.uint8,     │
                    │ contiguous)    │
                    └───────┬────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Dlib HOG + Linear SVM     │
              │  Face Detector             │
              │  detector(gray, upsample=1)│
              │  → [dlib.rectangle, ...]   │
              └─────────────┬──────────────┘
                            │  (for each face bounding box)
              ┌─────────────▼──────────────┐
              │  Dlib ERT Shape Predictor  │
              │  shape_predictor_68...dat  │
              │  predictor(gray, rect)     │
              │  → 68 × (x, y) coords     │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Visualization             │
              │  cv2.rectangle (bbox)      │
              │  cv2.circle × 68 (green)   │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Optional: EAR Computation │
              │  landmarks[36:42] → right  │
              │  landmarks[42:48] → left   │
              │  EAR < 0.2 → DROWSY alert  │
              └────────────────────────────┘
```

---

## 🏆 Results & Benchmarks

### Inference Timing — Dlib HOG + ERT across Resolutions (T4 GPU for resize, CPU for detection)

| Resolution  | Avg (ms) | Min (ms) | Notes                         |
|-------------|----------|----------|-------------------------------|
| 320×240     | < 2.0    | < 1.5    | Ultra-fast, small faces risk  |
| 480×360     | ~2–3     | ~1.5     | Good balance                  |
| 640×480     | ~3–5     | ~2.5     | **Recommended for real-time** |
| 1280×720    | ~8–12    | ~6       | HD; borderline 30fps          |
| 1920×1080   | ~18–25   | ~15      | Full HD; requires frame skip  |

> **Key finding:** Dlib's ERT algorithm lives up to its "One Millisecond" paper title at lower resolutions. At 480p, landmark inference is consistently sub-5ms on CPU — sufficient for smooth 30fps real-time performance. The HOG face detector is the actual bottleneck, not the landmark predictor.

---

## 🗺️ The 68 Landmark Map

| Region           | Indices   | Count | Key Use Cases                          |
|------------------|-----------|-------|----------------------------------------|
| Jaw line         | 0 – 16    | 17    | Face shape, jaw tracking               |
| Right eyebrow    | 17 – 21   | 5     | Surprise/anger expression              |
| Left eyebrow     | 22 – 26   | 5     | Surprise/anger expression              |
| Nose bridge      | 27 – 30   | 4     | Head pose estimation                   |
| Nose bottom      | 31 – 35   | 5     | Nostril detection                      |
| Right eye        | 36 – 41   | 6     | EAR, gaze tracking, blink detection    |
| Left eye         | 42 – 47   | 6     | EAR, gaze tracking, blink detection    |
| Outer lip        | 48 – 59   | 12    | Lip reading, MAR yawn detection        |
| Inner lip        | 60 – 67   | 8     | Mouth openness, speech sync            |

Standard: **iBUG 300-W** annotation scheme (Sagonas et al., 2013).

---

## 🔬 Extensions

### Extension 1 — Eye Aspect Ratio (EAR) · Drowsiness Detection

```
EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)
```

Using the 6 eye landmarks (p1–p6):
- **Open eye:** EAR ≈ 0.25–0.35
- **Blinking:** EAR drops briefly below 0.15 for 2–3 frames
- **Drowsy:** EAR sustained below 0.20 for > 1 second (≈ 30 frames at 30fps)

Implemented in `Section 7` of the notebook.

### Extension 2 — GPU Benchmark vs. MediaPipe Face Mesh (Section 8)

Compares Dlib ERT (CPU, 68 points) against MediaPipe Face Mesh (GPU, 468 points) across 5 resolutions. Includes GPU-accelerated image resizing via PyTorch bilinear interpolation on the T4.

### Extension 3 — Video File Processing (Section 6)

Full per-frame loop on an uploaded `.mp4` file — reads frames, runs detection + landmark prediction, writes annotated output video (`output_landmarks.mp4`).

---

## 📁 Project Structure

```
landmark-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── LandMark_Detection.ipynb     # Full pipeline (9 sections):
│                                    # 1. Environment setup
│                                    # 2. Model download
│                                    # 3. Core pipeline (detect_landmarks, draw_landmarks)
│                                    # 4. Demo 1 — Static image
│                                    # 5. Demo 2 — Colab webcam (JS bridge)
│                                    # 6. Demo 3 — Video file processing
│                                    # 7. EAR drowsiness detection
│                                    # 8. GPU benchmark vs. MediaPipe
│                                    # 9. Local deployment script export
│
├── app/
│   └── LandMarkDFacePoints.py       # Local real-time webcam app
│
├── output_samples/
│   └── README.md                    # Sample outputs
│
└── docs/
    └── landmark_map.md              # Detailed 68-point annotation reference
```

---

## ⚙️ Setup & Usage

### Option A — Google Colab (Recommended)

Open `notebooks/LandMark_Detection.ipynb` in [Google Colab](https://colab.research.google.com).

The notebook installs all dependencies automatically in Cell 1:

```bash
pip install numpy==1.26.4 scipy opencv-python-headless imutils dlib
```

> **Note:** `numpy<2` is required — Dlib's Python bindings have a known incompatibility with NumPy 2.x. The notebook pins `numpy==1.26.4`.

Run cells in order. Section 5 requires browser camera access permission.

---

### Option B — Local Setup

#### 1. Clone the repository

```bash
git clone https://github.com/prabhu-313/landmark-detection.git
cd landmark-detection
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Download the shape predictor model

The pre-trained model file (`shape_predictor_68_face_landmarks.dat`, ~97 MB) is **not included in the repo** due to size. Download it:

```bash
# Automated download (Python)
python -c "
import urllib.request, bz2, os
url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
urllib.request.urlretrieve(url, 'shape_predictor_68_face_landmarks.dat.bz2')
with bz2.BZ2File('shape_predictor_68_face_landmarks.dat.bz2') as f:
    open('shape_predictor_68_face_landmarks.dat', 'wb').write(f.read())
os.remove('shape_predictor_68_face_landmarks.dat.bz2')
print('Done.')
"
```

Or download directly from [dlib.net/files](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract.

Place the `.dat` file in the repo root (alongside `app/LandMarkDFacePoints.py`).

---

## 🖥 Local Deployment

Run the real-time webcam app:

```bash
cd app
python LandMarkDFacePoints.py
```

- Connects to your default webcam (`cv2.VideoCapture(0)`)
- Detects faces and overlays all 68 landmarks live at ~30fps
- Press **`q`** to quit

**Requirements for local deployment:** Physical webcam, Python 3.10+, all dependencies installed, `shape_predictor_68_face_landmarks.dat` in the working directory.

---

## 🔧 Key Design Decisions

| Decision | Rationale |
|---|---|
| `numpy==1.26.4` pin | Dlib C++ bindings fail with NumPy 2.x due to ABI changes in the array buffer protocol |
| `np.ascontiguousarray(gray, dtype=np.uint8)` | Dlib's HOG detector requires a C-contiguous memory layout; non-contiguous arrays (after slicing/resize) cause cryptic segfaults |
| `upsample=1` in detector | Upsamples input once before detection; finds faces down to ~80×80px at the cost of ~2x compute |
| `imutils.resize(width=500)` | Caps frame width for consistent processing speed; aspect ratio preserved automatically |
| Separate `detect_landmarks` / `draw_landmarks` functions | Clean separation of inference and visualization; makes the detection output reusable for EAR, head pose, etc. |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ==1.26.4 | Array ops; pinned for Dlib compatibility |
| `opencv-python` | 4.x | Video capture, image I/O, drawing |
| `dlib` | 19.x | Face detector + shape predictor |
| `imutils` | latest | Frame resize, shape_to_np utility |
| `scipy` | latest | EAR Euclidean distance computation |

---

## 📚 References

- Kazemi, V. & Sullivan, J. (2014). *One millisecond face alignment with an ensemble of regression trees.* CVPR.
- Sagonas, C. et al. (2013). *300 Faces In-The-Wild Challenge: The first facial landmark localization challenge.* ICCV Workshops. (iBUG 300-W dataset)
- Dalal, N. & Triggs, B. (2005). *Histograms of oriented gradients for human detection.* CVPR. (HOG descriptor)
- Soukupová, T. & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* (EAR formula)

---

## 👤 Author

**Prabhupada Samantaray** · [Gmail](mailto:psray313@gmail.com) · [LinkedIn](https://www.linkedin.com/in/prabhupada-samantaray-13apr2002/) · [GitHub](https://github.com/prabhu-313)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
