# Automatic License Plate Recognition (ALPR) System

## Overview
This project is an **Automatic License Plate Recognition (ALPR)** system that detects, extracts, and identifies license plates from images and video streams using **YOLOv5** and **OCR**.

## Features
- **License Plate Detection** using YOLOv5.
- **OCR (Optical Character Recognition)** using Tesseract.
- **Country Identification** based on regex patterns.
- **Real-Time Detection** via webcam.
- **Dataset Annotation & Conversion** from PascalVOC to YOLO format.

---

## Project Structure
```
ALPR-System/
│── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── annotations/
│   │   ├── train/
│   │   └── val/
│── yolov5/ (Cloned YOLOv5 repository)
│── models/
│   ├── best.pt (Trained YOLOv5 model)
│── src/
│   ├── annotation.py (Convert XML to YOLO format)
│   ├── detect.py (License plate detection)
│   ├── ocr.py (OCR & text extraction)
│   ├── country_identification.py (Identify country using regex)
│   ├── alpr_pipeline.py (Full pipeline integration)
│   ├── real_time.py (Real-time ALPR with webcam)
│── requirements.txt
│── license_plates.yaml (YOLO dataset configuration)
│── README.md
```

---

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ALPR-System.git
cd ALPR-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data Preparation
### 1. Annotate Images with LabelImg
```bash
pip install labelImg
labelImg  # Launch tool
```
- Draw bounding boxes around license plates.
- Save annotations in `PascalVOC (.xml)` format.

### 2. Convert Annotations to YOLO Format
```bash
python src/annotation.py
```

---

## Training YOLOv5 Model
### 1. Clone YOLOv5 Repository
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Configure Dataset
Modify `license_plates.yaml`:
```yaml
path: ../data
train: images/train
val: images/val
names:
  0: license_plate
```

### 3. Train the Model
```bash
python train.py --img 640 --batch 16 --epochs 50 --data ../license_plates.yaml --weights yolov5s.pt
```
- Trained weights are saved in `runs/train/exp/weights/best.pt`.

---

## License Plate Detection & OCR
### 1. Detect License Plates in an Image
```bash
python src/detect.py --image_path test_image.jpg --model_path models/best.pt
```

### 2. Full ALPR Pipeline
```bash
python src/alpr_pipeline.py --image_path test_image.jpg --model_path models/best.pt
```

### 3. Real-Time ALPR
```bash
python src/real_time.py --model_path models/best.pt
```
Press `q` to exit the webcam stream.

---

## Improvements & Future Work
- **Integrate EasyOCR** for better character recognition.
- **Improve post-processing** to reduce OCR errors.
- **Support multiple license plate formats** for different countries.
- **Train a custom OCR model** for improved accuracy.

---

## Requirements
```bash
# Install dependencies
torch>=1.10.0
torchvision>=0.11.1
opencv-python>=4.5.4
pytesseract>=0.3.8
numpy>=1.21.4
```

---

## Acknowledgments
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [LabelImg for annotations](https://github.com/heartexlabs/labelImg)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.