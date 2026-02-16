import sys
import os
import time
import cv2
import numpy as np

# --- 📦 AI & Image Processing Libs ---
from rembg import remove, new_session 
from ultralytics import YOLO 

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QLineEdit, QMessageBox, QSlider, QComboBox, 
                               QSizePolicy, QGroupBox, QFileDialog, QScrollArea)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# --- 🔌 Core Logic & Config ---
# ปรับ Path ให้ Python มองเห็น src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.config import MODE_PATHS
try:
    from src.core.camera import CameraManager
    from src.core.detector import ObjectDetector
    from src.core.image_utils import ImageProcessor
except ImportError as e:
    print(f"⚠️ Core modules not found: {e}")
    sys.exit(1)

# ลด Log
os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

class CollectorStation(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VisionForge: Dataset Collector (Multi-Modal)")
        self.resize(1400, 900)

        # --- 1. System Setup ---
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()
        
        # Default Mode = Pills
        self.current_mode = "pills"
        self.save_base_dir = MODE_PATHS["pills"]["raw_dir"]
        
        # Init AI
        try:
            self.rembg_session = new_session("u2net")
            print("✅ Rembg Ready")
        except:
            self.rembg_session = None

        # Variables
        self.is_recording = False
        self.count_saved = 0
        self.frame_count = 0
        self.current_zoom = 1.0

        # --- 2. UI Setup ---
        self.apply_styles()
        self.init_ui()

        # --- 3. Start Loop ---
        self.refresh_models()
        self.camera.start(0)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #f0f0f0; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
            QFrame.Sidebar { background-color: #252526; border-left: 1px solid #3e3e42; }
            QGroupBox { border: 1px solid #454545; border-radius: 6px; margin-top: 12px; font-weight: bold; color: #aaa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #3e3e42; border: 1px solid #555; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background-color: #4e4e52; }
            
            QPushButton#RecordBtn { background-color: #e74c3c; font-size: 16px; font-weight: bold; height: 45px; border: none; }
            QPushButton#RecordBtn:checked { background-color: #c0392b; }
            
            QLineEdit, QComboBox { background-color: #333337; border: 1px solid #555; padding: 6px; border-radius: 3px; color: white; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 6px; background: #1e1e1e; border-radius: 3px; }
            QSlider::handle:horizontal { background: #3b8ed0; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)

        # --- LEFT: Video ---
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #000;")
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(10,10,10,10)
        
        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_layout.addWidget(self.lbl_video)
        self.main_layout.addWidget(self.video_container, stretch=3)

        # --- RIGHT: Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setProperty("class", "Sidebar")
        self.sidebar.setFixedWidth(420)
        self.side_layout = QVBoxLayout(self.sidebar)
        self.side_layout.setContentsMargins(15, 15, 15, 15)

        # 1. Mode Selector (Pills vs Boxes)
        grp_mode = QGroupBox("🔄 COLLECTION MODE")
        l_mode = QVBoxLayout(grp_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["💊 PILLS MODE", "📦 BOXES MODE"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_change)
        l_mode.addWidget(self.combo_mode)
        
        # Path Display
        self.lbl_path = QLabel(f"Save to: .../raw_pills")
        self.lbl_path.setStyleSheet("color: #888; font-size: 12px;")
        l_mode.addWidget(self.lbl_path)
        self.side_layout.addWidget(grp_mode)

        # 2. System Control
        grp_sys = QGroupBox("🛠️ SYSTEM")
        l_sys = QVBoxLayout(grp_sys)
        
        # YOLO Model
        h1 = QHBoxLayout(); h1.addWidget(QLabel("YOLO:"))
        self.combo_model = QComboBox()
        self.combo_model.currentTextChanged.connect(self.load_model)
        h1.addWidget(self.combo_model, 1); l_sys.addLayout(h1)
        
        # Zoom
        h_zoom = QHBoxLayout()
        btn_zm = QPushButton("-"); btn_zm.setFixedWidth(35); btn_zm.clicked.connect(lambda: self.change_zoom(-1))
        btn_zp = QPushButton("+"); btn_zp.setFixedWidth(35); btn_zp.clicked.connect(lambda: self.change_zoom(1))
        self.lbl_zoom = QLabel("1.0x")
        h_zoom.addWidget(QLabel("Zoom:")); h_zoom.addWidget(btn_zm); h_zoom.addWidget(self.lbl_zoom); h_zoom.addWidget(btn_zp)
        l_sys.addLayout(h_zoom)
        
        # Brightness
        l_sys.addWidget(QLabel("Brightness:"))
        self.sl_bright = QSlider(Qt.Horizontal); self.sl_bright.setRange(-100, 100); self.sl_bright.setValue(0)
        l_sys.addWidget(self.sl_bright)
        self.side_layout.addWidget(grp_sys)

        # 3. Dataset Control
        grp_data = QGroupBox("📸 DATASET RECORDER")
        l_data = QVBoxLayout(grp_data)
        
        l_data.addWidget(QLabel("Class Name (Folder Name):"))
        self.entry_class = QLineEdit()
        self.entry_class.setPlaceholderText("e.g. paracetamol_500mg")
        l_data.addWidget(self.entry_class)
        
        # Interval
        h_int = QHBoxLayout()
        h_int.addWidget(QLabel("Save Every N Frames:"))
        self.lbl_interval = QLabel("5")
        self.sl_interval = QSlider(Qt.Horizontal); self.sl_interval.setRange(1, 30); self.sl_interval.setValue(5)
        self.sl_interval.valueChanged.connect(lambda v: self.lbl_interval.setText(str(v)))
        h_int.addWidget(self.sl_interval); h_int.addWidget(self.lbl_interval)
        l_data.addLayout(h_int)

        # Options
        self.combo_save_type = QComboBox()
        self.combo_save_type.addItems(["Save Raw Crop (Recommended)", "Save Rembg (Processed)"])
        l_data.addWidget(self.combo_save_type)
        
        self.side_layout.addWidget(grp_data)
        self.side_layout.addStretch()

        # Record Button
        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_record)
        self.side_layout.addWidget(self.btn_record)

        self.main_layout.addWidget(self.sidebar)

    # ================= FUNC =================
    def on_mode_change(self, index):
        if index == 0:
            self.current_mode = "pills"
        else:
            self.current_mode = "boxes"
        
        # 🔥 สลับ Path ตามโหมด
        self.save_base_dir = MODE_PATHS[self.current_mode]["raw_dir"]
        folder_name = os.path.basename(self.save_base_dir)
        self.lbl_path.setText(f"Save to: .../{folder_name}")
        
        # (Optional) โหลดโมเดล YOLO ประจำโหมดนั้นๆ ถ้ามีไฟล์อยู่จริง
        # model_name = MODE_PATHS[self.current_mode]["yolo_model"]
        # self.detector.load_model(os.path.join("models", model_name))

    def change_zoom(self, d):
        self.current_zoom = max(1.0, min(5.0, self.current_zoom + d))
        self.lbl_zoom.setText(f"{self.current_zoom:.1f}x")
        self.camera.set_zoom(self.current_zoom)

    def refresh_models(self):
        models = self.detector.get_available_models()
        self.combo_model.clear()
        if models:
            self.combo_model.addItems(models)
            # พยายามเลือก Model ให้ตรงกับ Mode (ถ้ามี)
            default_model = f"yolo_{self.current_mode}.pt"
            if default_model in models:
                self.combo_model.setCurrentText(default_model)

    def load_model(self, name):
        if name: self.detector.load_model(os.path.join("models", name))

    def toggle_record(self):
        if self.btn_record.isChecked():
            if not self.entry_class.text().strip():
                QMessageBox.warning(self, "Error", "❗ Please enter a Class Name first!")
                self.btn_record.setChecked(False)
                return
            
            self.is_recording = True
            self.count_saved = 0
            self.btn_record.setText("STOP RECORDING (0)")
            self.btn_record.setStyleSheet("background-color: #c0392b;") 
            self.entry_class.setEnabled(False)
            self.combo_mode.setEnabled(False)
        else:
            self.is_recording = False
            self.btn_record.setText("START RECORDING")
            self.btn_record.setStyleSheet("background-color: #e74c3c;") 
            self.entry_class.setEnabled(True)
            self.combo_mode.setEnabled(True)
            QMessageBox.information(self, "Done", f"Session Saved: {self.count_saved} images\nLocation: {self.save_base_dir}")

    # ================= CORE LOOP =================
    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None: return

        # 1. Image Process
        bright = self.sl_bright.value() / 100.0
        # Zoom 1.0 (Software) because Hardware zoom is handled
        processed = self.processor.apply_filters(frame, zoom=1.0, bright=bright)
        display = processed.copy()
        self.processor.draw_crosshair(display)

        # 2. YOLO Detect
        # (Collector มักจะใช้ Confidence ต่ำๆ เพื่อให้จับภาพได้เยอะๆ แล้วค่อยไปคัดออก)
        best_box = self.detector.predict(processed, conf=0.5)

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            
            # Draw Box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Recording Logic
            if self.is_recording:
                self.frame_count += 1
                if self.frame_count % self.sl_interval.value() == 0:
                    raw_crop = processed[y1:y2, x1:x2]
                    if raw_crop.size > 0:
                        self.save_dataset_image(raw_crop)

        self.show_video(display)

    def save_dataset_image(self, img_bgr):
        # สร้างโฟลเดอร์ตามชื่อยา (Class Name)
        class_name = self.entry_class.text().strip().replace(" ", "_")
        target_dir = os.path.join(self.save_base_dir, class_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # ตั้งชื่อไฟล์ด้วย Timestamp
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.png"
        full_path = os.path.join(target_dir, filename)

        # Save Logic
        save_mode = self.combo_save_type.currentIndex()
        
        if save_mode == 0: # Save Raw (Recommended)
            cv2.imwrite(full_path, img_bgr)
        
        elif save_mode == 1: # Save Rembg
            if self.rembg_session:
                try:
                    out = remove(img_bgr, session=self.rembg_session)
                    cv2.imwrite(full_path, out)
                except:
                    cv2.imwrite(full_path, img_bgr) # Fallback
            else:
                cv2.imwrite(full_path, img_bgr)

        self.count_saved += 1
        self.btn_record.setText(f"STOP RECORDING ({self.count_saved})")

    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CollectorStation()
    window.show()
    sys.exit(app.exec())