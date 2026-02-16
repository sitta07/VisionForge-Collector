import sys
import os
import time
import cv2
import numpy as np

# --- 📦 AI & Image Processing Libs ---
from rembg import remove, new_session 
from ultralytics import YOLO 

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QFrame, 
    QLineEdit, QMessageBox, QSlider, QComboBox, 
    QSizePolicy, QGroupBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# --- 🔌 Core Logic & Config ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.config import MODE_PATHS
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

        self.setWindowTitle("VisionForge: Dataset Collector (Multi-Modal)")
        self.resize(1400, 900)

        # =========================
        # 1️⃣ System Setup
        # =========================
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()

        self.current_mode = "pills"
        self.save_base_dir = MODE_PATHS["pills"]["raw_dir"]

        try:
            self.rembg_session = new_session("u2net")
            print("✅ Rembg Ready")
        except:
            self.rembg_session = None

        self.is_recording = False
        self.count_saved = 0
        self.frame_count = 0
        self.current_zoom = 1.0

        # =========================
        # 2️⃣ UI Setup
        # =========================
        self.apply_styles()
        self.init_ui()

        # =========================
        # 3️⃣ Start Loop
        # =========================
        self.refresh_models()
        self.camera.start(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # =====================================================
    # 🎨 UI STYLE
    # =====================================================
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #f0f0f0; font-family: 'Segoe UI'; font-size: 14px; }
            QFrame.Sidebar { background-color: #252526; border-left: 1px solid #3e3e42; }
            QGroupBox { border: 1px solid #454545; border-radius: 6px; margin-top: 12px; font-weight: bold; color: #aaa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #3e3e42; border: 1px solid #555; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background-color: #4e4e52; }
            QPushButton#RecordBtn { background-color: #e74c3c; font-size: 16px; height: 45px; border: none; }
            QPushButton#RecordBtn:checked { background-color: #c0392b; }
            QLineEdit, QComboBox { background-color: #333337; border: 1px solid #555; padding: 6px; border-radius: 3px; color: white; }
        """)

    # =====================================================
    # 🖥️ UI Layout
    # =====================================================
    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # -------- LEFT: VIDEO --------
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #000;")
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(10, 10, 10, 10)

        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_layout.addWidget(self.lbl_video)

        self.main_layout.addWidget(self.video_container, stretch=3)

        # -------- RIGHT: SIDEBAR --------
        self.sidebar = QFrame()
        self.sidebar.setProperty("class", "Sidebar")
        self.sidebar.setFixedWidth(420)
        self.side_layout = QVBoxLayout(self.sidebar)
        self.side_layout.setContentsMargins(15, 15, 15, 15)

        # Mode
        grp_mode = QGroupBox("COLLECTION MODE")
        l_mode = QVBoxLayout(grp_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["PILLS MODE", "BOXES MODE"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_change)
        l_mode.addWidget(self.combo_mode)

        self.lbl_path = QLabel("Save to: .../raw_pills")
        self.lbl_path.setStyleSheet("color: #888; font-size: 12px;")
        l_mode.addWidget(self.lbl_path)

        self.side_layout.addWidget(grp_mode)

        # Dataset
        grp_data = QGroupBox("DATASET RECORDER")
        l_data = QVBoxLayout(grp_data)

        l_data.addWidget(QLabel("Class Name:"))
        self.entry_class = QLineEdit()
        self.entry_class.setPlaceholderText("e.g. paracetamol_500mg")
        l_data.addWidget(self.entry_class)

        l_data.addWidget(QLabel("Save Every N Frames:"))
        self.sl_interval = QSlider(Qt.Horizontal)
        self.sl_interval.setRange(1, 30)
        self.sl_interval.setValue(5)
        l_data.addWidget(self.sl_interval)

        self.combo_save_type = QComboBox()
        self.combo_save_type.addItems([
            "Save Raw Crop",
            "Save Rembg (Processed)"
        ])
        l_data.addWidget(self.combo_save_type)

        self.side_layout.addWidget(grp_data)
        self.side_layout.addStretch()

        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_record)
        self.side_layout.addWidget(self.btn_record)

        self.main_layout.addWidget(self.sidebar)

    # =====================================================
    # 🔄 Mode Switch
    # =====================================================
    def on_mode_change(self, index):
        self.current_mode = "pills" if index == 0 else "boxes"
        self.save_base_dir = MODE_PATHS[self.current_mode]["raw_dir"]
        folder_name = os.path.basename(self.save_base_dir)
        self.lbl_path.setText(f"Save to: .../{folder_name}")

    # =====================================================
    # 🎥 Main Loop
    # =====================================================
    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return

        processed = self.processor.apply_filters(frame, zoom=1.0, bright=0)
        display = processed.copy()

        best_box = self.detector.predict(processed, conf=0.5)

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.is_recording:
                self.frame_count += 1
                if self.frame_count % self.sl_interval.value() == 0:
                    crop = processed[y1:y2, x1:x2]
                    if crop.size > 0:
                        self.save_dataset_image(crop)

        self.show_video(display)

    # =====================================================
    # 💾 Save Logic
    # =====================================================
    def save_dataset_image(self, img_bgr):
        class_name = self.entry_class.text().strip().replace(" ", "_")
        target_dir = os.path.join(self.save_base_dir, class_name)
        os.makedirs(target_dir, exist_ok=True)

        filename = f"{int(time.time()*1000)}.png"
        full_path = os.path.join(target_dir, filename)

        if self.combo_save_type.currentIndex() == 0:
            cv2.imwrite(full_path, img_bgr)
        else:
            if self.rembg_session:
                try:
                    out = remove(img_bgr, session=self.rembg_session)
                    cv2.imwrite(full_path, out)
                except:
                    cv2.imwrite(full_path, img_bgr)
            else:
                cv2.imwrite(full_path, img_bgr)

        self.count_saved += 1
        self.btn_record.setText(f"STOP RECORDING ({self.count_saved})")

    # =====================================================
    # ▶ Toggle Recording
    # =====================================================
    def toggle_record(self):
        if self.btn_record.isChecked():
            if not self.entry_class.text().strip():
                QMessageBox.warning(self, "Error", "Please enter Class Name!")
                self.btn_record.setChecked(False)
                return

            self.is_recording = True
            self.count_saved = 0
            self.btn_record.setText("STOP RECORDING (0)")
        else:
            self.is_recording = False
            self.btn_record.setText("START RECORDING")
            QMessageBox.information(self, "Done", f"Saved {self.count_saved} images.")

    # =====================================================
    # 🖥️ Display Frame
    # =====================================================
    def show_video(self, img):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.lbl_video.size(),
                Qt.KeepAspectRatio
            )
        )

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()


# =========================================================
# 🚀 RUN
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CollectorStation()
    window.show()
    sys.exit(app.exec())
