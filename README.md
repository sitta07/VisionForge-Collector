![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Qt](https://img.shields.io/badge/UI-PySide6%20(Qt)-green)
![License](https://img.shields.io/badge/License-Personal-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
# VisionForge Pro: Smart Dataset Collector

VisionForge Pro is a professional-grade dataset collection tool designed specifically for Computer Vision workflows. It supports ML engineers in capturing high-quality images of pills and small objects. The system includes built-in object detection for automatic cropping and supports hardware zoom control via the UVC standard.

## Features
- AI-powered detection using YOLOv8 for real-time object detection
- Hardware zoom control via UVC (supports levels 1–10, including -100 offset for YoloCam S3 compatibility)
- Pill Enhanced preset (LAB Color Space + CLAHE) to enhance pill imprints and surface texture without altering original colors
- High-performance UI built with PySide6 (Qt) with multi-threading support
- Automated recording system with structured folder management by class name

## System Requirements
- Operating System: Linux (Ubuntu 22.04 LTS recommended) or Windows 10/11
- Python: 3.10 or higher
- Dependencies:
  - PySide6
  - opencv-python-headless
  - ultralytics
  - numpy

## Installation

### 1. Install System Dependencies (Linux only)
```bash
sudo apt-get update
sudo apt-get install binutils libxcb-cursor0
```

### 2. Set Up Virtual Environment and Install Python Packages
```bash
python -m venv venv
source venv/bin/activate  # Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Prepare AI Models

Create a models folder in the project root:

```bash
mkdir models
```

Download YOLOv8n ONNX model:

```bash
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx
```

Alternatively, you may place any custom YOLOv8 ONNX model inside the models directory.

Expected structure:

```
VisionForge-Collector/
├── models/
│   ├── yolov8n.onnx
│   └── your_custom_model.onnx
```

The application automatically detects available models from this directory.

## Usage

### Run from Source Code
```bash
./run_app.sh
or
python src/main.py
```

### Interface Guide
- System Control: Select a YOLO model (.onnx) from the dropdown and use Force Autofocus if needed
- Image Settings: Adjust Zoom Level for hardware control and select "Pill Enhanced" preset when capturing pill surfaces
- Dataset Config: Enter class name and choose output directory
- Recording: Click Start Recording to automatically crop and save detected objects

## Project Structure
```
VisionForge-Collector/
├── src/
│   ├── main.py
│   ├── ui_qt.py
│   ├── camera.py
│   ├── detector.py
│   └── image_utils.py
├── models/
├── requirements.txt
├── README.md
└── run_app.sh
```

## Production Build

To build a standalone executable:

```bash
pyinstaller --name "VisionForgePro" \
            --windowed \
            --onefile \
            --clean \
            --paths src \
            src/main.py
```

After building, place the models folder at the same directory level as the VisionForgePro executable inside the dist folder.

## License

© 2026 Sitta Boonkaew. All rights reserved.

This project is a personal project.
