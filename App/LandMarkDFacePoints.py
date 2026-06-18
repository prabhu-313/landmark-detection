"""
LandMark: Real-Time Facial Landmark Detection
------------------------------------------------
Runs locally with a connected webcam.
Detects faces (Dlib HOG + SVM) and overlays 68 facial
landmarks (Dlib Ensemble of Regression Trees) on each
frame of a live video stream.
Requires: numpy<2 (pip install "numpy==1.26.4")
Usage:
    python LandMarkDFacePoints.py
    Press 'q' to quit.
"""

import cv2
import dlib
import imutils
import numpy as np
import time

PREDICTOR_PATH    = "shape_predictor_68_face_landmarks.dat"
FRAME_WIDTH       = 500
UPSAMPLE          = 1
EAR_THRESHOLD     = 0.20
EAR_CONSEC_FRAMES = 20

FACIAL_LANDMARKS_IDXS = {
    "jaw":           (0,  17),
    "right_eyebrow": (17, 22),
    "left_eyebrow":  (22, 27),
    "nose":          (27, 36),
    "right_eye":     (36, 42),
    "left_eye":      (42, 48),
    "mouth":         (48, 68),
}

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_aspect_ratio(eye_pts):
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    ear_counter = 0
    fps_counter = 0
    fps_start   = time.time()
    display_fps = 0.0

    print("[INFO] LandMark started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying...")
            continue

        frame = imutils.resize(frame, width=FRAME_WIDTH)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = np.ascontiguousarray(gray, dtype=np.uint8)
        rects = detector(gray, UPSAMPLE)

        for rect in rects:
            shape     = predictor(gray, rect)
            landmarks = shape_to_np(shape)

            x, y = rect.left(), rect.top()
            w, h = rect.width(), rect.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for (px, py) in landmarks:
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

            left_eye  = landmarks[42:48]
            right_eye = landmarks[36:42]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            if ear < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSY!", (10, frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                ear_counter = 0

            cv2.putText(frame, f"EAR: {ear:.3f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            display_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start   = time.time()

        cv2.putText(frame, f"FPS: {display_fps:.1f}  Faces: {len(rects)}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("LandMark — Facial Landmarks  [q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] LandMark stopped.")


if __name__ == "__main__":
    main()
