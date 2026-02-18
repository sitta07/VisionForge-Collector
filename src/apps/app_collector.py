import sys
import os
import time
import cv2
import numpy as np
import subprocess  # <--- เพิ่มตัวนี้สำหรับยิงคำสั่ง Linux Command

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
# (ปรับ Path ตาม Structure โปรเจกต์จริงของคุณ)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Config & Core Modules
# ถ้าหาไม่เจอ ให้ Comment ส่วนนี้ แล้วใช้ Mock Class แทนเพื่อ Test UI
try:
    from src.core.config import MODE_PATHS, BASE_DIR
    from src.core.camera import CameraManager
    from src.core.detector import ObjectDetector
    from src.core.image_utils import ImageProcessor
except ImportError:
    # ⚠️ Mockup Classes กรณีรัน Test แล้วไม่มี Modules จริง
    print("⚠️ Core modules not found. Using Mock classes.")
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
        def remove_background(self, i): return i

os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

class CollectorStation(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VisionForge: Dataset Collector (Hardware Zoom Edition)")
        self.resize(1400, 900)

        # --- 1. System Setup ---
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()
        
        # Default Mode
        self.current_mode = "pills"
        self.current_config = MODE_PATHS["pills"]
        self.save_base_dir = self.current_config["raw_dir"]
        
        # AI Init (โหลด Rembg รอไว้)
        try:
            self.rembg_session = new_session("u2net")
            print("✅ Rembg Engine Ready")
        except:
            self.rembg_session = None

        # Variables
        self.is_recording = False
        self.count_saved = 0
        self.frame_count = 0
        
        # Config Hardware Zoom Defaults
        self.zoom_min = 50
        self.zoom_max = 98
        self.zoom_default = 50
        self.camera_device = "/dev/video0" # ⚠️ เช็คด้วยว่ากล้องคุณคือ video0 ไหม

        # --- 2. UI & Start ---
        self.apply_styles()
        self.init_ui()
        
        # เริ่มต้นโหลด Model
        self.refresh_models()
        self.camera.start(0)
        
        # Reset Zoom to Default at startup
        self.set_hardware_zoom(self.zoom_default)

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
        self.lbl_video = QLabel("Initializing Camera...")
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

        # 2. Preview Card
        grp_prev = QGroupBox("👁️ LIVE PREVIEW")
        l_prev = QVBoxLayout(grp_prev)
        self.lbl_preview = QLabel("Waiting...")
        self.lbl_preview.setFixedSize(200, 160)
        self.lbl_preview.setStyleSheet("border: 1px dashed #555; background-color: #111;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        l_prev.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        self.side_layout.addWidget(grp_prev)

        # 3. System Controls (Hardware Zoom Here!)
        grp_sys = QGroupBox("🛠️ SYSTEM CONTROL")
        l_sys = QVBoxLayout(grp_sys)
        
        # YOLO Selection
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

        # 🔥 HARDWARE ZOOM SLIDER 🔥
        l_sys.addWidget(QLabel("📸 Hardware Zoom (v4l2):"))
        h_zoom = QHBoxLayout()
        
        self.sl_zoom = QSlider(Qt.Horizontal)
        self.sl_zoom.setRange(self.zoom_min, self.zoom_max)
        self.sl_zoom.setValue(self.zoom_default)
        self.sl_zoom.setTickPosition(QSlider.TicksBelow)
        self.sl_zoom.setTickInterval(5)
        
        # เชื่อม Event: valueChanged = เปลี่ยนทันทีที่ลาก (Real-time)
        self.sl_zoom.valueChanged.connect(self.set_hardware_zoom)
        
        self.lbl_zoom_val = QLabel(str(self.zoom_default))
        self.lbl_zoom_val.setFixedWidth(35)
        self.lbl_zoom_val.setAlignment(Qt.AlignCenter)
        self.lbl_zoom_val.setStyleSheet("font-weight: bold; color: #4cd137;")

        h_zoom.addWidget(self.sl_zoom)
        h_zoom.addWidget(self.lbl_zoom_val)
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

    # ================= LOGIC: HARDWARE ZOOM (NEW) =================
    def set_hardware_zoom(self, value):
        """
        สั่ง Driver กล้องโดยตรงผ่าน v4l2-ctl
        """
        # อัปเดตตัวเลขหน้า UI
        self.lbl_zoom_val.setText(str(value))
        
        # สร้างคำสั่ง Terminal
        # เทียบเท่า: v4l2-ctl -d /dev/video0 --set-ctrl zoom_absolute=XX
        cmd = [
            "v4l2-ctl", 
            "-d", self.camera_device, 
            "--set-ctrl", f"zoom_absolute={value}"
        ]
        
        try:
            # ใช้ subprocess.run เพื่อยิงคำสั่ง
            # check=False เพื่อไม่ให้โปรแกรม Crash ถ้ากล้องไม่ตอบสนอง
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            print("⚠️ Error: ไม่พบคำสั่ง v4l2-ctl (ลองรัน: sudo apt install v4l-utils)")
        except Exception as e:
            print(f"⚠️ Zoom Error: {e}")

    # ================= LOGIC: MODEL & MODE =================
    def on_mode_change(self, index):
        self.current_mode = "pills" if index == 0 else "boxes"
        self.current_config = MODE_PATHS[self.current_mode]
        
        self.save_base_dir = self.current_config["raw_dir"]
        self.lbl_path.setText(f"Save to: .../{os.path.basename(self.save_base_dir)}")
        
        target_model = self.current_config["yolo_model"]
        idx = self.combo_model.findText(target_model)
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)

    def refresh_models(self):
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
            self.combo_model.addItems(sorted(list(set(found_models))))
            idx = self.combo_model.findText(self.current_config["yolo_model"])
            if idx >= 0: self.combo_model.setCurrentIndex(idx)
            else: self.combo_model.setCurrentIndex(0)
            self.load_model(self.combo_model.currentText())
        else:
            self.combo_model.addItem("No models found")
        self.combo_model.blockSignals(False)

    def load_model(self, name):
        if not name or "No models" in name: return
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

        # Note: เราไม่ต้องทำ Software Zoom ใน ImageProcessor แล้ว
        # เพราะ Hardware กล้องส่งภาพที่ซูมแล้วมาให้เลย
        processed = self.processor.apply_filters(frame, zoom=1.0) 
        display = processed.copy()
        self.processor.draw_crosshair(display)

        # Detect
        best_box = self.detector.predict(processed, conf=0.5)

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            raw_crop = processed[y1:y2, x1:x2]
            
            if raw_crop.size > 0:
                preview_img = self.processor.remove_background(raw_crop)
                self.show_preview(preview_img)
                
                if self.is_recording:
                    self.frame_count += 1
                    if self.frame_count % self.sl_interval.value() == 0:
                        self.save_image(raw_crop)

        self.show_video(display)

    def save_image(self, img_bgr):
        cname = self.entry_class.text().strip().replace(" ", "_")
        target_dir = os.path.join(self.save_base_dir, cname)
        os.makedirs(target_dir, exist_ok=True)
        
        filename = f"{int(time.time()*1000)}.png"
        full_path = os.path.join(target_dir, filename)
        
        if self.combo_save.currentIndex() == 0: # Save AI Rembg
            if self.rembg_session:
                try:
                    out = remove(img_bgr, session=self.rembg_session)
                    cv2.imwrite(full_path, out)
                except:
                    cv2.imwrite(full_path, img_bgr)
            else:
                cv2.imwrite(full_path, img_bgr)
        else:
            cv2.imwrite(full_path, img_bgr)
            
        self.count_saved += 1
        self.btn_record.setText(f"STOP RECORDING ({self.count_saved})")

    # ================= HELPERS =================
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
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
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
            QGroupBox { border: 1px solid #444; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QSlider::groove:horizontal { border: 1px solid #999; height: 8px; background: #333; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #4cd137; border: 1px solid #5c5c5c; width: 18px; margin: -2px 0; border-radius: 9px; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CollectorStation()
    w.show()
    sys.exit(app.exec())