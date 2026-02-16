import sys
import os
import time
import cv2
import numpy as np

# --- 📦 AI Libs ---
from rembg import remove, new_session 
from ultralytics import YOLO 

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QLineEdit, QMessageBox, QSlider, QComboBox, 
                               QSizePolicy, QGroupBox, QFileDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# 🔌 CORE SETUP
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Config & Core Modules
from src.core.config import MODE_PATHS, BASE_DIR
try:
    from src.core.camera import CameraManager
    from src.core.detector import ObjectDetector
    from src.core.image_utils import ImageProcessor
except ImportError as e:
    print(f"⚠️ Core modules not found: {e}")
    sys.exit(1)

os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

class CollectorStation(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VisionForge: Dataset Collector (Hybrid Engine)")
        self.resize(1400, 900)

        # --- 1. System Setup ---
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()
        
        # Default Mode
        self.current_mode = "pills"
        self.current_config = MODE_PATHS["pills"]
        self.save_base_dir = self.current_config["raw_dir"]
        
        # AI Init (โหลด Rembg รอไว้สำหรับการ Save)
        try:
            self.rembg_session = new_session("u2net")
            print("✅ Rembg Engine Ready (for saving)")
        except:
            self.rembg_session = None

        # Variables
        self.is_recording = False
        self.count_saved = 0
        self.frame_count = 0
        self.current_zoom = 1.0

        # --- 2. UI & Start ---
        self.apply_styles()
        self.init_ui()
        
        # เริ่มต้นโหลด Model ตามโหมดปัจจุบัน
        self.refresh_models()
        self.camera.start(0)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0)

        # --- LEFT: Video ---
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #000;")
        v_layout = QVBoxLayout(self.video_container)
        self.lbl_video = QLabel("Initializing...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        v_layout.addWidget(self.lbl_video)
        self.main_layout.addWidget(self.video_container, stretch=3)

        # --- RIGHT: Sidebar ---
        self.sidebar = QFrame(); self.sidebar.setFixedWidth(420)
        self.sidebar.setProperty("class", "Sidebar")
        self.side_layout = QVBoxLayout(self.sidebar)
        
        # 1. Mode Selector
        grp_mode = QGroupBox("🔄 COLLECTION MODE")
        l_mode = QVBoxLayout(grp_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["💊 PILLS MODE", "📦 BOXES MODE"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_change)
        l_mode.addWidget(self.combo_mode)
        
        self.lbl_path = QLabel(f"Save to: .../{os.path.basename(self.save_base_dir)}")
        self.lbl_path.setStyleSheet("color: #888; font-size: 11px;")
        l_mode.addWidget(self.lbl_path)
        self.side_layout.addWidget(grp_mode)

        # 🔥 2. Preview Card (เพิ่มกลับมาแล้ว!)
        grp_prev = QGroupBox("👁️ LIVE PREVIEW (Fast Cut)")
        l_prev = QVBoxLayout(grp_prev)
        self.lbl_preview = QLabel("Waiting...")
        self.lbl_preview.setFixedSize(200, 160)
        self.lbl_preview.setStyleSheet("border: 1px dashed #555; background-color: #111;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        l_prev.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        self.side_layout.addWidget(grp_prev)

        # 3. System Controls
        grp_sys = QGroupBox("🛠️ SYSTEM")
        l_sys = QVBoxLayout(grp_sys)
        
        # 🔥 YOLO Selection (เพิ่มมาใหม่!)
        l_sys.addWidget(QLabel("YOLO Model:"))
        h_yolo = QHBoxLayout()
        self.combo_model = QComboBox()
        self.combo_model.currentTextChanged.connect(self.load_model)
        btn_browse = QPushButton("📂")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self.browse_model)
        h_yolo.addWidget(self.combo_model, 1)
        h_yolo.addWidget(btn_browse)
        l_sys.addLayout(h_yolo)

        # Zoom
        h_zoom = QHBoxLayout()
        h_zoom.addWidget(QLabel("Zoom:"))
        btn_out = QPushButton("-"); btn_out.setFixedSize(35, 30); btn_out.clicked.connect(lambda: self.change_zoom(-0.5))
        self.lbl_zoom = QLabel("1.0x"); self.lbl_zoom.setFixedWidth(40); self.lbl_zoom.setAlignment(Qt.AlignCenter)
        btn_in = QPushButton("+"); btn_in.setFixedSize(35, 30); btn_in.clicked.connect(lambda: self.change_zoom(0.5))
        h_zoom.addWidget(btn_out); h_zoom.addWidget(self.lbl_zoom); h_zoom.addWidget(btn_in)
        l_sys.addLayout(h_zoom)
        
        self.side_layout.addWidget(grp_sys)

        # 4. Dataset Settings
        grp_data = QGroupBox("📸 RECORDER SETTINGS")
        l_data = QVBoxLayout(grp_data)
        
        l_data.addWidget(QLabel("Class Name (Folder):"))
        self.entry_class = QLineEdit()
        self.entry_class.setPlaceholderText("e.g. paracetamol_500mg")
        l_data.addWidget(self.entry_class)
        
        h_int = QHBoxLayout()
        h_int.addWidget(QLabel("Save Every (Frames):"))
        self.sl_interval = QSlider(Qt.Horizontal); self.sl_interval.setRange(1, 30); self.sl_interval.setValue(5)
        self.lbl_interval = QLabel("5")
        self.sl_interval.valueChanged.connect(lambda v: self.lbl_interval.setText(str(v)))
        h_int.addWidget(self.sl_interval); h_int.addWidget(self.lbl_interval)
        l_data.addLayout(h_int)
        
        self.combo_save = QComboBox()
        self.combo_save.addItems(["Save AI Rembg (High Quality)", "Save Raw Crop"])
        l_data.addWidget(self.combo_save)
        
        self.side_layout.addWidget(grp_data)
        self.side_layout.addStretch()

        # RECORD BUTTON
        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.setCheckable(True)
        self.btn_record.setFixedHeight(50)
        self.btn_record.clicked.connect(self.toggle_record)
        self.side_layout.addWidget(self.btn_record)

        self.main_layout.addWidget(self.sidebar)

    # ================= LOGIC: MODEL & MODE =================
    def on_mode_change(self, index):
        self.current_mode = "pills" if index == 0 else "boxes"
        self.current_config = MODE_PATHS[self.current_mode]
        
        # 1. Change Path
        self.save_base_dir = self.current_config["raw_dir"]
        self.lbl_path.setText(f"Save to: .../{os.path.basename(self.save_base_dir)}")
        
        # 2. Auto Select Model (ถ้ามีใน list)
        target_model = self.current_config["yolo_model"]
        idx = self.combo_model.findText(target_model)
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)

    def refresh_models(self):
        # Scan หาไฟล์ .pt / .onnx ในโฟลเดอร์ model_weights หรือ models
        search_dirs = [
            os.path.join(BASE_DIR, "model_weights"),
            os.path.join(BASE_DIR, "models")
        ]
        
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        
        found_models = []
        for d in search_dirs:
            if os.path.exists(d):
                files = [f for f in os.listdir(d) if f.endswith(('.pt', '.onnx'))]
                found_models.extend(files)
        
        if found_models:
            self.combo_model.addItems(sorted(list(set(found_models)))) # Unique & Sorted
            # Try set default
            default = self.current_config["yolo_model"]
            idx = self.combo_model.findText(default)
            if idx >= 0: self.combo_model.setCurrentIndex(idx)
            else: self.combo_model.setCurrentIndex(0)
            
            # Load ตัวแรกทันที
            self.load_model(self.combo_model.currentText())
        else:
            self.combo_model.addItem("No models found")
            
        self.combo_model.blockSignals(False)

    def load_model(self, name):
        if not name or "No models" in name: return
        
        # หา path จริง
        model_path = None
        for d in ["model_weights", "models"]:
            p = os.path.join(BASE_DIR, d, name)
            if os.path.exists(p):
                model_path = p
                break
        
        if model_path:
            self.detector.load_model(model_path)
            print(f"👁️ Loaded YOLO: {name}")

    def browse_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select YOLO", BASE_DIR, "Models (*.pt *.onnx)")
        if fname:
            name = os.path.basename(fname)
            self.combo_model.addItem(name)
            self.combo_model.setCurrentText(name)
            self.detector.load_model(fname)

    # ================= LOGIC: MAIN LOOP =================
    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None: return

        processed = self.processor.apply_filters(frame, zoom=1.0)
        display = processed.copy()
        self.processor.draw_crosshair(display)

        # Detect
        best_box = self.detector.predict(processed, conf=0.5)

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            
            # วาดกรอบ
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract Crop
            raw_crop = processed[y1:y2, x1:x2]
            
            if raw_crop.size > 0:
                # 🔥 1. FAST PREVIEW (ใช้ OpenCV ตัดแบบเร็วๆ โชว์บนจอ)
                # วิธีนี้จะไม่หน่วง UI เหมือน Rembg
                preview_img = self.processor.remove_background(raw_crop)
                self.show_preview(preview_img)
                
                # 🔥 2. RECORDING LOGIC (High Quality Save)
                if self.is_recording:
                    self.frame_count += 1
                    if self.frame_count % self.sl_interval.value() == 0:
                        self.save_image(raw_crop)

        self.show_video(display)

    def save_image(self, img_bgr):
        # Create Folder
        cname = self.entry_class.text().strip().replace(" ", "_")
        target_dir = os.path.join(self.save_base_dir, cname)
        os.makedirs(target_dir, exist_ok=True)
        
        filename = f"{int(time.time()*1000)}.png"
        full_path = os.path.join(target_dir, filename)
        
        # 🔥 HIGH QUALITY SAVE (AI Rembg)
        if self.combo_save.currentIndex() == 0: # Save AI Rembg
            if self.rembg_session:
                try:
                    # AI ตัดพื้นหลัง (ช้าหน่อยแต่เนียน)
                    out = remove(img_bgr, session=self.rembg_session)
                    cv2.imwrite(full_path, out)
                except:
                    cv2.imwrite(full_path, img_bgr) # Fallback
            else:
                cv2.imwrite(full_path, img_bgr)
        else:
            # Save Raw (สำหรับกล่องยา หรือคนชอบรูปดิบ)
            cv2.imwrite(full_path, img_bgr)
            
        self.count_saved += 1
        self.btn_record.setText(f"STOP RECORDING ({self.count_saved})")

    # ================= HELPERS =================
    def change_zoom(self, delta):
        new_zoom = max(1.0, min(5.0, self.current_zoom + delta))
        if new_zoom != self.current_zoom:
            self.current_zoom = new_zoom
            self.lbl_zoom.setText(f"{self.current_zoom:.1f}x")
            self.camera.set_zoom(self.current_zoom)

    def toggle_record(self):
        if self.btn_record.isChecked():
            if not self.entry_class.text().strip():
                QMessageBox.warning(self, "Error", "❗ Please enter Class Name first!")
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
            QMessageBox.information(self, "Saved", f"Session Saved: {self.count_saved} images")

    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))
    
    def show_preview(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped() # OpenCV Preview is RGB (usually)
        # ถ้าภาพที่ส่งมามี Alpha (4 channels)
        if c == 4:
             qi = QImage(img.data, w, h, c*w, QImage.Format_RGBA8888)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio))

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #f0f0f0; font-family: 'Segoe UI'; font-size: 14px; }
            QLineEdit, QComboBox { background-color: #333; padding: 5px; border: 1px solid #555; }
            QPushButton { background-color: #444; padding: 8px; border-radius: 4px; }
            QPushButton#RecordBtn { background-color: #e74c3c; font-weight: bold; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CollectorStation()
    w.show()
    sys.exit(app.exec())