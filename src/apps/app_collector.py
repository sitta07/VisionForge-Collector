import sys
import os
import time
import cv2
import numpy as np
import subprocess

# --- 📦 AI Libs (No Rembg anymore) ---
from ultralytics import YOLO 

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QLineEdit, QMessageBox, QSlider, QComboBox, 
                               QSizePolicy, QGroupBox, QFileDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

# 🔌 CORE SETUP
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Try Import or Mock
try:
    from src.core.config import MODE_PATHS, BASE_DIR
    from src.core.camera import CameraManager
    from src.core.detector import ObjectDetector
    from src.core.image_utils import ImageProcessor
except ImportError:
    print("⚠️ Core modules not found. Using Mock classes for UI testing.")
    BASE_DIR = os.getcwd()
    MODE_PATHS = {
        "pills": {"raw_dir": "./data/pills", "yolo_model": "yolov8n.pt"},
        "boxes": {"raw_dir": "./data/boxes", "yolo_model": "yolov8n.pt"}
    }
    class CameraManager:
        def start(self, id): pass
        def get_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8)
        def set_zoom(self, z): pass
    class ObjectDetector:
        def load_model(self, p): pass
        def predict(self, i, conf): return None
    class ImageProcessor:
        def apply_filters(self, i, zoom): return i
        def draw_crosshair(self, i): pass

os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

class CollectorStation(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VisionForge: Dataset Collector (Pro Edition)")
        self.resize(1400, 950)

        # --- 1. System Setup ---
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()
        
        # Default Config
        self.current_mode = "pills"
        self.current_config = MODE_PATHS["pills"]
        self.save_base_dir = self.current_config["raw_dir"]
        
        # Variables
        self.is_recording = False
        self.count_saved = 0
        self.frame_count = 0
        
        # Settings Defaults
        self.conf_threshold = 0.50  # Default Confidence
        self.zoom_min = 50
        self.zoom_max = 98
        self.zoom_default = 50
        self.camera_device = "/dev/video0"

        # --- 2. UI & Start ---
        self.apply_styles() # ✨ Load Aesthetic CSS
        self.init_ui()
        
        # Init System
        self.refresh_models()
        self.camera.start(0)
        self.set_hardware_zoom(self.zoom_default)

        # Timer Loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(15)

        # --- LEFT: Video Feed ---
        self.video_container = QFrame()
        self.video_container.setObjectName("VideoFrame")
        v_layout = QVBoxLayout(self.video_container)
        v_layout.setContentsMargins(0,0,0,0)
        
        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #000; border-radius: 8px;")
        v_layout.addWidget(self.lbl_video)
        self.main_layout.addWidget(self.video_container, stretch=3)

        # --- RIGHT: Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(420)
        self.sidebar.setObjectName("Sidebar")
        self.side_layout = QVBoxLayout(self.sidebar)
        self.side_layout.setContentsMargins(15, 15, 15, 15)
        self.side_layout.setSpacing(15)
        
        # 1. Header & Mode
        grp_mode = QGroupBox("📌 COLLECTION MODE")
        l_mode = QVBoxLayout(grp_mode)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["💊 PILLS DATASET", "📦 BOXES DATASET"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_change)
        l_mode.addWidget(self.combo_mode)
        
        self.lbl_path = QLabel(f"📂 Save to: .../{os.path.basename(self.save_base_dir)}")
        self.lbl_path.setObjectName("PathLabel")
        l_mode.addWidget(self.lbl_path)
        self.side_layout.addWidget(grp_mode)

        # 2. Preview Card
        grp_prev = QGroupBox("👁️ LIVE CROP PREVIEW")
        l_prev = QVBoxLayout(grp_prev)
        self.lbl_preview = QLabel("Waiting for object...")
        self.lbl_preview.setFixedSize(220, 180)
        self.lbl_preview.setObjectName("PreviewBox")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        l_prev.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        self.side_layout.addWidget(grp_prev)

        # 3. AI & Hardware Controls
        grp_sys = QGroupBox("⚙️ ENGINE CONTROLS")
        l_sys = QVBoxLayout(grp_sys)
        l_sys.setSpacing(12)
        
        # [A] YOLO Model
        h_yolo = QHBoxLayout()
        self.combo_model = QComboBox()
        self.combo_model.currentTextChanged.connect(self.load_model)
        btn_browse = QPushButton("📂")
        btn_browse.setFixedWidth(40)
        btn_browse.clicked.connect(self.browse_model)
        h_yolo.addWidget(self.combo_model, 1)
        h_yolo.addWidget(btn_browse)
        l_sys.addWidget(QLabel("🧠 YOLO Model:"))
        l_sys.addLayout(h_yolo)

        # [B] CONFIDENCE SLIDER (NEW! ✨)
        h_conf = QHBoxLayout()
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(1, 99) # 0.01 to 0.99
        self.sl_conf.setValue(50)    # Default 0.50
        self.sl_conf.valueChanged.connect(self.change_conf)
        
        self.lbl_conf = QLabel("0.50")
        self.lbl_conf.setFixedWidth(40)
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setStyleSheet("color: #00E676; font-weight: bold;")
        
        h_conf.addWidget(self.sl_conf)
        h_conf.addWidget(self.lbl_conf)
        l_sys.addWidget(QLabel("🎯 Confidence Threshold:"))
        l_sys.addLayout(h_conf)

        # [C] HARDWARE ZOOM
        h_zoom = QHBoxLayout()
        self.sl_zoom = QSlider(Qt.Horizontal)
        self.sl_zoom.setRange(self.zoom_min, self.zoom_max)
        self.sl_zoom.setValue(self.zoom_default)
        self.sl_zoom.valueChanged.connect(self.set_hardware_zoom)
        
        self.lbl_zoom_val = QLabel(str(self.zoom_default))
        self.lbl_zoom_val.setFixedWidth(40)
        self.lbl_zoom_val.setAlignment(Qt.AlignCenter)
        self.lbl_zoom_val.setStyleSheet("color: #29b6f6; font-weight: bold;")

        h_zoom.addWidget(self.sl_zoom)
        h_zoom.addWidget(self.lbl_zoom_val)
        l_sys.addWidget(QLabel("🔍 Hardware Zoom (v4l2):"))
        l_sys.addLayout(h_zoom)
        
        self.side_layout.addWidget(grp_sys)

        # 4. Recording Settings
        grp_data = QGroupBox("📸 CAPTURE SETTINGS")
        l_data = QVBoxLayout(grp_data)
        
        self.entry_class = QLineEdit()
        self.entry_class.setPlaceholderText("🏷️ Enter Class Name (e.g. para_500)")
        l_data.addWidget(self.entry_class)
        
        h_int = QHBoxLayout()
        self.sl_interval = QSlider(Qt.Horizontal)
        self.sl_interval.setRange(1, 30)
        self.sl_interval.setValue(5)
        self.lbl_interval = QLabel("Every 5 Frames")
        self.sl_interval.valueChanged.connect(lambda v: self.lbl_interval.setText(f"Every {v} Frames"))
        
        h_int.addWidget(self.sl_interval)
        h_int.addWidget(self.lbl_interval)
        l_data.addLayout(h_int)
        
        self.side_layout.addWidget(grp_data)
        self.side_layout.addStretch()

        # RECORD BUTTON
        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.setCheckable(True)
        self.btn_record.setFixedHeight(55)
        self.btn_record.setCursor(Qt.PointingHandCursor)
        self.btn_record.clicked.connect(self.toggle_record)
        self.side_layout.addWidget(self.btn_record)

        self.main_layout.addWidget(self.sidebar)

    # ================= LOGIC: CONFIDENCE =================
    def change_conf(self, value):
        self.conf_threshold = value / 100.0
        self.lbl_conf.setText(f"{self.conf_threshold:.2f}")

    # ================= LOGIC: HARDWARE ZOOM =================
    def set_hardware_zoom(self, value):
        self.lbl_zoom_val.setText(str(value))
        cmd = ["v4l2-ctl", "-d", self.camera_device, "--set-ctrl", f"zoom_absolute={value}"]
        try:
            subprocess.run(cmd, check=False)
        except:
            pass # Silent fail if no v4l2

    # ================= LOGIC: MAIN LOOP =================
    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None: return

        processed = self.processor.apply_filters(frame, zoom=1.0)
        display = processed.copy()
        self.processor.draw_crosshair(display)

        # 🔥 Use Dynamic Confidence
        best_box = self.detector.predict(processed, conf=self.conf_threshold)

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            
            # Draw aesthetic box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.putText(display, f"{best_box.conf[0]:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
            
            # Extract Raw Crop
            raw_crop = processed[y1:y2, x1:x2]
            
            if raw_crop.size > 0:
                self.show_preview(raw_crop) # Just Raw Preview
                
                if self.is_recording:
                    self.frame_count += 1
                    if self.frame_count % self.sl_interval.value() == 0:
                        self.save_image(raw_crop)

        self.show_video(display)

    def save_image(self, img_bgr):
        # Simple Raw Save Logic
        cname = self.entry_class.text().strip().replace(" ", "_")
        target_dir = os.path.join(self.save_base_dir, cname)
        os.makedirs(target_dir, exist_ok=True)
        
        filename = f"{int(time.time()*1000)}.png"
        full_path = os.path.join(target_dir, filename)
        
        cv2.imwrite(full_path, img_bgr) # Save Raw Only
            
        self.count_saved += 1
        self.btn_record.setText(f"STOP RECORDING ({self.count_saved})")

    # ================= HELPERS & SETUP =================
    def on_mode_change(self, index):
        self.current_mode = "pills" if index == 0 else "boxes"
        self.current_config = MODE_PATHS[self.current_mode]
        self.save_base_dir = self.current_config["raw_dir"]
        self.lbl_path.setText(f"📂 Save to: .../{os.path.basename(self.save_base_dir)}")
        
        target = self.current_config["yolo_model"]
        if self.combo_model.findText(target) >= 0:
            self.combo_model.setCurrentText(target)

    def refresh_models(self):
        # (Same logic as before, simplified for brevity)
        search_dirs = [os.path.join(BASE_DIR, "model_weights"), os.path.join(BASE_DIR, "models")]
        self.combo_model.clear()
        found = []
        for d in search_dirs:
            if os.path.exists(d):
                found.extend([f for f in os.listdir(d) if f.endswith(('.pt', '.onnx'))])
        if found:
            self.combo_model.addItems(sorted(list(set(found))))
            self.load_model(self.combo_model.currentText())
        else:
            self.combo_model.addItem("No models found")

    def load_model(self, name):
        if not name or "No models" in name: return
        for d in ["model_weights", "models"]:
            p = os.path.join(BASE_DIR, d, name)
            if os.path.exists(p):
                self.detector.load_model(p)
                print(f"🧠 AI Loaded: {name}")
                break

    def browse_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select YOLO", BASE_DIR, "Models (*.pt *.onnx)")
        if fname:
            self.combo_model.addItem(os.path.basename(fname))
            self.combo_model.setCurrentText(os.path.basename(fname))

    def toggle_record(self):
        if self.btn_record.isChecked():
            if not self.entry_class.text().strip():
                QMessageBox.warning(self, "Missing Info", "⚠️ Please enter Class Name first!")
                self.btn_record.setChecked(False)
                return
            self.is_recording = True
            self.count_saved = 0
            self.btn_record.setText("STOP RECORDING (0)")
            self.btn_record.setStyleSheet("""
                QPushButton#RecordBtn { 
                    background-color: #ff4757; 
                    border: 1px solid #ff6b81;
                    color: white; 
                }
            """)
            self.entry_class.setEnabled(False)
            self.combo_mode.setEnabled(False)
        else:
            self.is_recording = False
            self.btn_record.setText("START RECORDING")
            self.btn_record.setStyleSheet("") # Revert to default ID style
            self.entry_class.setEnabled(True)
            self.combo_mode.setEnabled(True)
            QMessageBox.information(self, "Saved", f"✅ Session Complete: {self.count_saved} images saved.")

    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))
    
    def show_preview(self, img):
        if not img.flags['C_CONTIGUOUS']:
            img = img.copy()

        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_preview.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio))

    def apply_styles(self):
        # 🎨 Aesthetic Dark Theme CSS
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            
            /* Panels */
            QFrame#Sidebar { background-color: #1e1e1e; border-left: 1px solid #333; }
            QFrame#VideoFrame { background-color: #000; border: 1px solid #333; border-radius: 8px; }
            
            /* Group Box */
            QGroupBox { 
                border: 1px solid #333; 
                border-radius: 6px; 
                margin-top: 10px; 
                padding-top: 15px; 
                background-color: #252526;
                font-weight: bold;
                color: #aaa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            
            /* Inputs */
            QLineEdit, QComboBox { 
                background-color: #333; 
                padding: 8px; 
                border: 1px solid #444; 
                border-radius: 4px; 
                color: white;
            }
            QLineEdit:focus, QComboBox:focus { border: 1px solid #00E676; }
            
            /* Buttons */
            QPushButton { 
                background-color: #3a3a3a; 
                border: 1px solid #555; 
                padding: 8px; 
                border-radius: 4px; 
                color: white;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            
            /* Record Button */
            QPushButton#RecordBtn { 
                background-color: #00E676; 
                color: #121212; 
                font-weight: bold; 
                font-size: 14px;
                border: none;
            }
            QPushButton#RecordBtn:hover { background-color: #00C853; }
            
            /* Sliders */
            QSlider::groove:horizontal { 
                border: 1px solid #333; 
                height: 6px; 
                background: #2d2d2d; 
                margin: 2px 0; 
                border-radius: 3px; 
            }
            QSlider::handle:horizontal { 
                background: #00E676; 
                border: 1px solid #00C853; 
                width: 16px; 
                margin: -5px 0; 
                border-radius: 8px; 
            }
            
            /* Labels */
            QLabel#PathLabel { color: #666; font-size: 11px; }
            QLabel#PreviewBox { border: 2px dashed #444; background-color: #1a1a1a; border-radius: 6px; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CollectorStation()
    w.show()
    sys.exit(app.exec())