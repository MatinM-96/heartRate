# Remote Heart Rate Monitor using Computer Vision

This project implements a non-contact heart rate monitoring system using a webcam. It estimates heart rate (BPM) by analyzing skin color changes caused by blood flow using remote photoplethysmography (rPPG).

---

## 🔧 Setup

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔹 1. `hearRateOffline.py` (Robust, Multi-ROI POS version)

This version offers higher accuracy and uses multiple ROIs on the face. It processes RGB signals with the POS algorithm and filters noise using a Butterworth filter.

### ▶️ How to Run:
```bash
python hearRateOffline.py
```

📌 Features:
- Multi-ROI: Forehead + cheeks
- POS algorithm + Butterworth bandpass filter
- Welch PSD for BPM estimation
- SNR-based ROI selection
- CSV output (`bpm_log.csv`)

---

## 🔹 2. `hearRateOnline.py` (Lightweight, Real-Time version)

This version is designed for real-time performance. It uses Gaussian pyramids and FFT to extract the heart rate directly from facial video.

### ▶️ How to Run:
```bash
python hearRateOnline.py
```

📌 Features:
- Real-time heart rate estimation
- Gaussian pyramid (Level 3)
- FFT-based frequency analysis
- Live BPM plotting and CSV logging

---

## 📁 Output

Both scripts generate:

```text
bpm_log.csv
```

With format:
```
Time (s), BPM
```

---

## 📦 Method Summary

- **Face Detection:** `cvzone.FaceDetector`
- **Signal Extraction:** RGB from forehead + cheeks
- **Offline Method:** POS + Welch PSD (accurate but slower)
- **Online Method:** Gaussian Pyramid + FFT (real-time)
- **Display:** Real-time BPM with `cvzone.LivePlot`

---

## 🧪 Testing Results

- Accuracy within ±2 BPM of Apple Watch in best conditions
- Performs well under soft indoor lighting
- Removing eyeglasses improved results by reducing glare

---

## 👨‍💻 Authors

- **Matin Mohammadi**
- **Sander Gnanavel**

IKT452 – Remote Photoplethysmography Project, Spring 2025

---
