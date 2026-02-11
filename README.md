# 📸 SnapSet-Pro: AI Dataset Collector

**SnapSet-Pro** is a professional GUI tool for collecting and annotating computer vision datasets. 
Built with Python and Tkinter, optimized for creating YOLO datasets efficiently.

## ✨ Features
- **GUI Controls:** Zoom, Confidence Threshold, and Class Naming.
- **Preset System:** Save configurations for different objects.
- **Auto-Organization:** Images are saved into labeled folders automatically.
- **Portable:** Ready to be built into a standalone .exe.

## 🚀 Quick Start
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the app:
   `python src/main.py`

## 🛠 Building .exe
`pyinstaller --onefile --windowed --name="SnapSet-Pro" src/main.py`