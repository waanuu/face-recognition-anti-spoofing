# Face Recognition and Face Anti-Spoofing System

## Introduction

This project presents a real-time identity verification system that combines Face Recognition and Face Anti-Spoofing technologies. The system is designed to recognize authorized users while preventing spoofing attacks such as printed photos, replay attacks on mobile devices, and fake face images.

The project was developed as a Graduation Thesis in Artificial Intelligence and Data Science.

## Features

* Real-time face detection using SCRFD
* Face anti-spoofing using MobileNetV3
* Face recognition using ArcFace
* Fast identity search using FAISS
* Webcam-based authentication
* Real-time identity verification
* Protection against photo and screen replay attacks

## System Pipeline

Input Image / Webcam

↓

Face Detection (SCRFD)

↓

Face Anti-Spoofing (MobileNetV3)

↓

Face Alignment

↓

Feature Extraction (ArcFace)

↓

Similarity Search (FAISS)

↓

Identity Verification

## Technologies

* Python
* PyTorch
* OpenCV
* ONNX Runtime
* SCRFD
* MobileNetV3
* ArcFace
* FAISS

## Dataset

### Face Anti-Spoofing

Datasets used:

* LCC_FASD
* NUAA Imposter Database
* Self-collected Real/Fake Face Dataset

Processed images:

* Real Faces: 6,106
* Fake Faces: 11,247
* Total: 17,353

### Face Recognition

Datasets used:

* Celebrity Faces Dataset
* AsianCeleb Dataset
* Self-collected Vietnamese Celebrity Dataset

All face images were detected and aligned using SCRFD before training and evaluation.

## Experimental Results

### Face Anti-Spoofing Performance

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 97.69% |
| AUC-ROC  | 0.9971 |
| FAR      | 1.91%  |
| FRR      | 3.03%  |
| F1-Score | 0.9673 |

### Face Recognition Performance

| Metric                | Value  |
| --------------------- | ------ |
| Verification Accuracy | 93.00% |
| AUC                   | 0.9647 |
| EER                   | 8.87%  |

## Demo

### Real User Recognition

<img src="[Ảnh chụp màn hình 2026-05-15 175513.png](https://github.com/waanuu/face-recognition-anti-spoofing/blob/main/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-05-15%20175513.png)" width="700">

### Spoof Detection

<img src="[screenshots/demo_fake.png](https://github.com/waanuu/face-recognition-anti-spoofing/blob/main/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-05-15%20180136.png)" width="700">

### Identity Verification

<img src="[screenshots/demo_verify.png](https://github.com/waanuu/face-recognition-anti-spoofing/blob/main/%E1%BA%A2nh%20ch%E1%BB%A5p%20m%C3%A0n%20h%C3%ACnh%202026-05-15%20221117.png)" width="700">

Demo Video:

https://drive.google.com/drive/folders/1GQhI3oxqKWowVsDKzrbemaVsizMAJKnC?usp=sharing

## Installation

```bash
git clone https://github.com/waanuu/face-recognition-anti-spoofing.git

cd face-recognition-anti-spoofing

pip install -r requirements.txt
```

## Run

```bash
python app.py
```
