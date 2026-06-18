# 🗺️ The 68-Point Facial Landmark Map

## Standard: iBUG 300-W Annotation Scheme

The 68-point scheme used by Dlib's `shape_predictor_68_face_landmarks.dat` follows the **iBUG 300-W dataset** annotation standard, introduced in:

> Sagonas, C., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2013).  
> *300 Faces In-The-Wild Challenge: The first facial landmark localization challenge.*  
> ICCV Workshops.

---

## Point Index Reference

### Jaw Line (0 – 16) — 17 points

Points trace the jaw from right ear to left ear across the chin.

```
0  → rightmost jaw point (near right ear)
8  → chin tip (bottom centre)
16 → leftmost jaw point (near left ear)
```

**Use cases:** Face boundary, face shape classification, head tilt detection.

---

### Right Eyebrow (17 – 21) — 5 points

Traced from outer edge (right side of face) inward toward nose.

```
17 → outer corner of right eyebrow
21 → inner corner of right eyebrow (toward nose)
```

**Use cases:** Eyebrow raise detection (surprise, fear), anger (brow lowering).

---

### Left Eyebrow (22 – 26) — 5 points

Traced from inner edge (toward nose) outward.

```
22 → inner corner of left eyebrow (toward nose)
26 → outer corner of left eyebrow
```

**Use cases:** Same as right eyebrow; bilateral asymmetry analysis.

---

### Nose Bridge (27 – 30) — 4 points

Vertical points down the bridge of the nose.

```
27 → nose root (between eyebrows)
28 → upper nose bridge
29 → lower nose bridge
30 → nose tip
```

**Use cases:** Head pose estimation (point 30 is the primary "nose tip" for solvePnP).

---

### Nose Bottom (31 – 35) — 5 points

Horizontal points across the bottom of the nose.

```
31 → left nostril outer edge
33 → nose bottom centre
35 → right nostril outer edge
```

**Use cases:** Nose width measurement, nostril tracking.

---

### Right Eye (36 – 41) — 6 points

Clockwise from outer (right side) corner.

```
36 → outer corner (right side of face)
37 → upper eyelid, outer
38 → upper eyelid, inner
39 → inner corner (toward nose)
40 → lower eyelid, inner
41 → lower eyelid, outer
```

**Use cases:** Eye Aspect Ratio (EAR), blink detection, gaze estimation.

#### EAR Formula for Right Eye:
```
EAR_right = (‖p37−p41‖ + ‖p38−p40‖) / (2 · ‖p36−p39‖)
           = (‖pts[1]−pts[5]‖ + ‖pts[2]−pts[4]‖) / (2 · ‖pts[0]−pts[3]‖)
where pts = landmarks[36:42]
```

---

### Left Eye (42 – 47) — 6 points

Clockwise from inner (toward nose) corner.

```
42 → inner corner (toward nose)
43 → upper eyelid, inner
44 → upper eyelid, outer
45 → outer corner (left side of face)
46 → lower eyelid, outer
47 → lower eyelid, inner
```

**Use cases:** EAR, blink detection, gaze estimation.

#### EAR Formula for Left Eye:
```
EAR_left = (‖pts[1]−pts[5]‖ + ‖pts[2]−pts[4]‖) / (2 · ‖pts[0]−pts[3]‖)
where pts = landmarks[42:48]
```

---

### Outer Lip (48 – 59) — 12 points

Clockwise from left corner of mouth.

```
48 → left corner of mouth
51 → upper lip centre
54 → right corner of mouth
57 → lower lip centre
```

**Use cases:** Mouth Aspect Ratio (MAR), lip reading, smile detection, speech sync.

#### MAR Formula:
```
MAR = (‖p50−p58‖ + ‖p51−p57‖ + ‖p52−p56‖) / (3 · ‖p48−p54‖)
```

---

### Inner Lip (60 – 67) — 8 points

Inner boundary of the mouth, clockwise from left inner corner.

```
60 → left inner corner
62 → upper inner lip centre
64 → right inner corner
66 → lower inner lip centre
```

**Use cases:** Mouth openness, yawn detection, speech visibility estimation.

---

## FACIAL_LANDMARKS_IDXS Dictionary

```python
FACIAL_LANDMARKS_IDXS = {
    "jaw":            (0,  17),
    "right_eyebrow":  (17, 22),
    "left_eyebrow":   (22, 27),
    "nose":           (27, 36),
    "right_eye":      (36, 42),
    "left_eye":       (42, 48),
    "mouth":          (48, 68),
}
```

Usage:
```python
# Extract just the eye regions
right_eye_pts = landmarks[36:42]   # shape: (6, 2)
left_eye_pts  = landmarks[42:48]   # shape: (6, 2)
nose_tip      = landmarks[30]      # shape: (2,) — single point
```

---

## Coordinate System

All landmark coordinates are in OpenCV's pixel coordinate system:
- **Origin:** Top-left corner of the image
- **x:** Increases left → right
- **y:** Increases top → bottom
- **Units:** Pixels (integer, after `shape_to_np`)

When converting to normalized coordinates (for resolution-invariant computations):
```python
# Normalize by face bounding box dimensions
x_norm = (x - rect.left()) / rect.width()
y_norm = (y - rect.top()) / rect.height()
```

---

## Key Landmark Indices for Common Applications

| Application | Landmarks Used | Description |
|---|---|---|
| Eye blink / drowsiness | 36–47 | Both eyes (6 pts each) for EAR |
| Smile detection | 48, 54, 51, 57 | Mouth corners + top/bottom |
| Head pose (solvePnP) | 30, 8, 36, 45, 48, 54 | Nose tip, chin, eye corners, mouth corners |
| Gaze tracking | 36–47 + iris (if available) | Eye corners + pupil center |
| Face alignment | 36, 45 | Eye centers for rotation |
| Jaw clenching | 0–16 | Temporal movement of jaw contour |
| Eyebrow raise | 17–26 vs 36–47 | Vertical distance brow-to-eye |
