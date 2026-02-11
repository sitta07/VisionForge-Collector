# VisionForge Pro: Smart Dataset Collector

VisionForge Pro is a professional-grade dataset collection tool designed specifically for Computer Vision workflows. It supports ML engineers in capturing high-quality images of pills and small objects. The system includes built-in AI detection for automatic cropping and supports hardware zoom control via the UVC standard.

## 🛠 Features
- **AI-Powered Detection:** Uses YOLOv8 for real-time object detection.
- **Hardware Zoom Control:** Controls hardware zoom via UVC. Supports levels 1–10 (including -100 offset for YoloCam S3 compatibility).
- **Pill Enhanced Preset:** Advanced image processing mode (LAB Color Space + CLAHE) to enhance pill imprints and surface texture without altering original colors.
- **High-Fidelity UI:** Built with PySide6 (Qt) with multi-threading support for smooth video performance.
- **Production Recording:** Automated image capture with structured folder management based on class name.

## System Requirements
- **Operating System:** Linux (Ubuntu 22.04 LTS recommended) or Windows 10/11
- **Python:** 3.10 or higher
- **Dependencies:**
  - PySide6 (UI framework)
  - opencv-python-headless (Image processing)
  - ultralytics (YOLOv8 core)
  - numpy (Data processing)

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

## 🖥 Usage

### Run from Source Code
```bash
./run_app.sh
or
python src/main.py 
```

### Interface Guide
- **System Control:** Select a YOLO (.pt) model from the dropdown and click Force Autofocus if the image is blurry.
- **Image Settings:** Adjust Zoom Level (1–10) for hardware control. Select "Pill Enhanced" preset for capturing pills with imprints.
- **Dataset Config:** Enter a class name and choose an output directory.
- **Recording:** Click Start Recording. The system will automatically crop and save detected objects.

## Project Structure
```
VisionForge-Collector/
├── src/
│   ├── main.py          # Application entry point
│   ├── ui_qt.py         # UI logic (PySide6)
│   ├── camera.py        # Hardware interface (UVC/OpenCV)
│   ├── detector.py      # AI inference (YOLOv8)
│   └── image_utils.py   # Image processing algorithms
├── models/              # YOLO model files (.pt)
├── requirements.txt     # Required Python libraries
├── README.md            # Documentation
└── run_app.sh           # Production launcher (Linux)
```

## Production Build (Executable)

To build a standalone executable for desktop use without running through VS Code:
```bash
pyinstaller --name "VisionForgePro" \
            --windowed \
            --onefile \
            --clean \
            --paths src \
            src/main.py
```

After a successful build, place the `models/` folder at the same directory level as the `VisionForgePro` executable inside the `dist/` folder.

## 📄 License

© 2025 Sitta Boonkaew. All rights reserved.

This project is a personal project .