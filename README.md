# Automatic License Plate Recognition (ALPR) System

## Overview
This project is an **Automatic License Plate Recognition (ALPR)** system that detects, extracts, and identifies license plates from images and video streams using **YOLOv5** and **OCR**.

## Features
- **License Plate Detection** using YOLOv5.
- **OCR (Optical Character Recognition)** using Tesseract.
- **Country Identification** based on regex patterns.
- **Real-Time Detection** via webcam and YouTube video streams.
- **Dataset Annotation & Conversion** from PascalVOC to YOLO format.
- **Training YOLOv5 Model** with custom datasets.
- **Evaluation of Detection Accuracy** across multiple countries.


---

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ALPR-System.git
cd ALPR-System
```

---

## Data Preparation
### Annotate Images with Label-Studio or LabelImg
```bash
pip install labelImg
labelImg  # Launch tool
```
- Draw bounding boxes around license plates.
- Save annotations in YOLO format.

---

## Training YOLOv5 Model
### 1. Clone YOLOv5 Repository
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data ../data-yolo.yaml --weights yolov5s.pt
```
- Trained weights are saved in `yolov5/runs/train/exp/weights/best.pt`.

---

## Video Detection

```bash
python src/real_time.py --model_path models/best.pt
```
Press `q` to exit the webcam stream.

---

## Improvements & Future Work
- **Newer version of YOLO or trying other libraries** can increase accuracy of license plate labelling.
- **More images can be provided** for better plate recognition among different environments.
- **Different Filters** for better character recognition.
- **Tried EasyOCR, didnt work but maybe different scenarios** for better character recognition.
- **Improve post-processing** to reduce OCR errors.
- **Support multiple license plate formats** for different countries.
- **Train a custom OCR model** for improved accuracy.

---

## Requirements
```bash
# DO NOT USE EasyOCR with Tesseract at the same environment!
torch>=1.10.0
torchvision>=0.11.1
opencv-python>=4.5.4
pytesseract>=0.3.8
numpy>=1.21.4
yt_dlp
pafy
```

---

## Acknowledgments
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [LabelImg for annotations](https://github.com/heartexlabs/labelImg)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.