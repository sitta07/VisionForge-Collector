import sys
import os
import time
import cv2
import numpy as np

# --- 📦 AI & Image Processing Libs ---
# ต้องลง: pip install rembg[gpu] onnxruntime-gpu
from rembg import remove, new_session 
import onnxruntime as ort

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QComboBox, QSlider, QLineEdit, QScrollArea, 
                               QFileDialog, QMessageBox, QGroupBox, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QSize, Slot
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# --- 🔌 Import Backend Logic (เรียกใช้ไฟล์ที่คุณมีอยู่แล้ว) ---
try:
    from camera import CameraManager
    from detector import ObjectDetector
    from image_utils import ImageProcessor
    import config
    DEFAULT_PATH = config.DEFAULT_DATA_ROOT
except ImportError:
    print("⚠️ Error: ไม่พบไฟล์ Backend (camera.py, detector.py, image_utils.py)")
    print("กรุณาตรวจสอบว่าไฟล์เหล่านี้อยู่ในโฟลเดอร์เดียวกับ main.py")
    sys.exit(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VisionForge: Smart Collector Pro (Hybrid Engine)")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)


        # --- 1. System Setup ---
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()
        
        print("🔍 Checking AI Hardware Acceleration...")
        print(f"👉 Device: {ort.get_device()}") 
        print(f"👉 Providers Available: {ort.get_available_providers()}")
        # --- 2. AI Engine Setup (Rembg) ---
        # โหลดโมเดล AI ลง GPU รอไว้เลย (กันกระตุกตอนกด Save ครั้งแรก)
        print("⏳ Initializing Rembg (AI Background Remover)...")
        try:
            # 'u2net' คือโมเดลมาตรฐาน (สมดุลระหว่างความเร็วและความเนียน)
            # ถ้าต้องการความเร็วสูงสุดให้เปลี่ยนเป็น 'u2netp' (แต่ความเนียนจะลดลงนิดหน่อย)
            self.rembg_session = new_session("u2net") 
            print("✅ Rembg Ready! (GPU Engine Activated)")
        except Exception as e:
            print(f"❌ Rembg Init Failed: {e}")
            self.rembg_session = None
        
        # --- 3. Variable Initialization ---
        self.is_recording = False
        self.save_dir = DEFAULT_PATH
        self.count_saved = 0
        self.frame_count = 0
        self.current_zoom = 1 # เริ่มต้นที่ 1x

        # --- 4. UI Setup ---
        self.apply_styles()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.setup_video_panel()
        self.setup_sidebar()

        # --- 5. Start Loop ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.refresh_models()
        self.camera.start(0)
        self.timer.start(30) # 30ms ~ 33 FPS

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QFrame#Sidebar { background-color: #1e1e1e; border-left: 1px solid #333; }
            QFrame.Card { background-color: #252526; border-radius: 8px; border: 1px solid #333; }
            QLabel.CardTitle { color: #3b8ed0; font-weight: bold; font-size: 14px; }
            
            QPushButton { 
                background-color: #3a3a3a; border: none; border-radius: 4px; 
                padding: 8px; color: white; font-weight: bold;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
            
            QPushButton.ZoomBtn { 
                background-color: #2980b9; font-size: 18px; font-weight: bold; width: 40px;
            }
            QPushButton.ZoomBtn:hover { background-color: #3498db; }
            
            QPushButton#RecordBtn { 
                background-color: #e74c3c; font-size: 16px; height: 50px; 
            }
            QPushButton#RecordBtn:checked { background-color: #c0392b; }
            
            QLineEdit, QComboBox { 
                background-color: #1e1e1e; border: 1px solid #444; 
                border-radius: 4px; padding: 5px; color: white; 
            }
            QSlider::groove:horizontal {
                border: 1px solid #333; height: 6px; background: #1e1e1e; margin: 2px 0; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3b8ed0; border: 1px solid #3b8ed0; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
            }
        """)

    def setup_video_panel(self):
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #000;")
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_layout.addWidget(self.lbl_video)
        self.main_layout.addWidget(self.video_container, stretch=3)

    def setup_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(400)
        self.sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(15, 20, 15, 20)
        self.sidebar_layout.setSpacing(15)

        header = QLabel("VISION FORGE PRO")
        header.setStyleSheet("font-size: 24px; font-weight: 900; color: white; letter-spacing: 2px;")
        self.sidebar_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0,0,0,0)
        self.scroll_layout.setSpacing(15)
        scroll.setWidget(self.scroll_content)
        
        self.sidebar_layout.addWidget(scroll)

        self.create_preview_card()
        self.create_system_card()
        self.create_image_card()
        self.create_recording_card()

        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_record)
        self.sidebar_layout.addWidget(self.btn_record)
        self.sidebar_layout.setStretch(1, 1)
        self.main_layout.addWidget(self.sidebar)

    def create_card(self, title):
        card = QFrame()
        card.setProperty("class", "Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        lbl_title = QLabel(title)
        lbl_title.setProperty("class", "CardTitle")
        layout.addWidget(lbl_title)
        self.scroll_layout.addWidget(card)
        return layout

    def create_preview_card(self):
        layout = self.create_card("LIVE CROP PREVIEW")
        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(250, 200)
        self.lbl_preview.setStyleSheet("background-color: black; border: 1px solid #444; border-radius: 4px;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.addWidget(self.lbl_preview)
        layout.addWidget(container)

    def create_system_card(self):
        layout = self.create_card("SYSTEM CONTROL")
        layout.addWidget(QLabel("AI Model:"))
        self.combo_model = QComboBox()
        self.combo_model.currentTextChanged.connect(self.load_model)
        layout.addWidget(self.combo_model)
        btn_focus = QPushButton("⚡ FORCE AUTOFOCUS")
        btn_focus.clicked.connect(lambda: self.camera.trigger_autofocus())
        layout.addWidget(btn_focus)

    def create_image_card(self):
        layout = self.create_card("IMAGE SETTINGS")
        
        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("Preset Mode:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["Default", "Pill Enhanced (Texture)"])
        row_preset.addWidget(self.combo_preset, 1)
        layout.addLayout(row_preset)
        
        # --- ZOOM Controls ---
        row_zoom = QHBoxLayout()
        row_zoom.addWidget(QLabel("Zoom Level:"))
        
        btn_minus = QPushButton("-")
        btn_minus.setProperty("class", "ZoomBtn")
        btn_minus.setFixedSize(40, 35)
        btn_minus.clicked.connect(lambda: self.change_zoom(-1))
        
        self.lbl_zoom_val = QLabel("IN") if self.current_zoom == 1 else QLabel("OUT")
        self.lbl_zoom_val.setAlignment(Qt.AlignCenter)
        self.lbl_zoom_val.setStyleSheet("font-size: 16px; font-weight: bold; color: #3b8ed0;")
        self.lbl_zoom_val.setFixedWidth(60)
        
        btn_plus = QPushButton("+")
        btn_plus.setProperty("class", "ZoomBtn")
        btn_plus.setFixedSize(40, 35)
        btn_plus.clicked.connect(lambda: self.change_zoom(1))
        
        row_zoom.addStretch()
        row_zoom.addWidget(btn_minus)
        row_zoom.addWidget(self.lbl_zoom_val)
        row_zoom.addWidget(btn_plus)
        layout.addLayout(row_zoom)

        # Brightness/Contrast
        def add_slider(label, min_val, max_val, default):
            row = QHBoxLayout()
            lbl = QLabel(label)
            val_lbl = QLabel(f"{default:.1f}")
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val_lbl)
            layout.addLayout(row)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(min_val*10), int(max_val*10))
            slider.setValue(int(default*10))
            slider.valueChanged.connect(lambda v: val_lbl.setText(f"{v/10:.1f}"))
            layout.addWidget(slider)
            return slider

        self.sl_bright = add_slider("Brightness", -100, 100, 0)
        self.sl_cont = add_slider("Contrast", 0.5, 3.0, 1.0)

    def change_zoom(self, delta):
        # Toggle Zoom Logic (1 <-> 2)
        if self.current_zoom == 1:
            self.current_zoom = 2
            self.lbl_zoom_val.setText("OUT") # Zoom 2x (ซูมเข้า)
        else:
            self.current_zoom = 1
            self.lbl_zoom_val.setText("IN") # Zoom 1x (ปกติ)
        
        # ส่งค่าไปที่ Hardware Camera (สำคัญ!)
        # หมายเหตุ: ใน camera.py ต้องมีฟังก์ชัน set_zoom(val) รองรับ
        self.camera.set_zoom(self.current_zoom)

    def create_recording_card(self):
        layout = self.create_card("DATASET CONFIG")
        layout.addWidget(QLabel("Class Name:"))
        self.entry_class = QLineEdit()
        self.entry_class.setPlaceholderText("e.g. pill_type_a")
        layout.addWidget(self.entry_class)
        
        path_row = QHBoxLayout()
        self.lbl_path = QLabel(f".../{os.path.basename(self.save_dir)}")
        self.lbl_path.setStyleSheet("color: gray; font-size: 11px;")
        btn_browse = QPushButton("📂")
        btn_browse.setFixedWidth(40)
        btn_browse.clicked.connect(self.browse_folder)
        path_row.addWidget(self.lbl_path)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)
        
        layout.addWidget(QLabel("Save Every N Frames:"))
        self.sl_interval = QSlider(Qt.Horizontal)
        self.sl_interval.setRange(1, 60)
        self.sl_interval.setValue(5)
        self.lbl_interval_val = QLabel("5")
        self.sl_interval.valueChanged.connect(lambda v: self.lbl_interval_val.setText(str(v)))
        int_row = QHBoxLayout()
        int_row.addWidget(self.sl_interval)
        int_row.addWidget(self.lbl_interval_val)
        layout.addLayout(int_row)
        
        layout.addWidget(QLabel("AI Confidence:"))
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(1, 100)
        self.sl_conf.setValue(60)
        layout.addWidget(self.sl_conf)

    # ==========================================
    # 🔥 CORE LOGIC: UPDATE FRAME
    # ==========================================
    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None: return

        # 1. Image Enhancement
        bright_val = self.sl_bright.value() / 10.0
        cont_val = self.sl_cont.value() / 10.0
        preset_name = self.combo_preset.currentText()

        # Hardware Zoom ถูกจัดการโดย self.camera.set_zoom() แล้ว
        # Software Zoom ปรับเป็น 1.0 (ปิดไว้) เพื่อไม่ให้ภาพแตก
        processed = self.processor.apply_filters(
            frame, 
            zoom=1.0, 
            bright=bright_val, 
            contrast=cont_val,
            preset=preset_name
        )
        
        display = processed.copy()
        self.processor.draw_crosshair(display)

        # 2. AI Detection (YOLO)
        conf_val = self.sl_conf.value() / 100.0
        best_box = self.detector.predict(processed, conf=conf_val)
        
        # ตัวแปรเก็บภาพสำหรับ Preview (ใช้ OpenCV)
        preview_crop = None
        
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cls_id = int(best_box.cls[0])
            conf = best_box.conf[0].item()
            name = self.detector.model.names.get(cls_id, f"ID_{cls_id}")
            
            # วาดกรอบบนหน้าจอหลัก
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{name} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Crop ภาพดิบออกมา (ยังไม่ลบพื้นหลัง)
            raw_crop = processed[y1:y2, x1:x2]
            
            if raw_crop.size > 0:
                # [Fast Path] สร้างภาพ Preview ด้วย OpenCV (เร็ว แต่ไม่เนียน 100%)
                # ใช้เพื่อโชว์บนมุมขวาบนเท่านั้น เพื่อไม่ให้ UI หน่วง
                preview_crop = self.processor.remove_background(raw_crop)
                
                # [Quality Path] ถ้ากดอัด (Recording) ให้ส่งภาพดิบไปลบด้วย AI
                if self.is_recording:
                    self.frame_count += 1
                    if self.frame_count % self.sl_interval.value() == 0:
                        self.save_image_with_rembg(raw_crop)

        self.show_video(display)
        self.show_preview(preview_crop)

    def show_video(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        p = QPixmap.fromImage(convert_to_Qt_format)
        lbl_h = self.lbl_video.height()
        lbl_w = self.lbl_video.width()
        self.lbl_video.setPixmap(p.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio))

    def show_preview(self, cv_img):
        if cv_img is None:
            self.lbl_preview.setText("WAITING...")
            self.lbl_preview.setPixmap(QPixmap())
            return
        # แปลง BGR -> RGB สำหรับแสดงผล
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qimg).scaled(250, 200, Qt.KeepAspectRatio))

    # ==========================================
    # 💾 AI SAVING LOGIC (Rembg)
    # ==========================================
    def save_image_with_rembg(self, img_bgr):
        cname = self.entry_class.text().strip() or "unknown"
        path = os.path.join(self.save_dir, cname)
        os.makedirs(path, exist_ok=True)
        fname = f"{int(time.time()*1000)}.png"
        full_path = os.path.join(path, fname)

        # ถ้า Rembg พร้อมใช้งาน ให้ใช้ AI ลบพื้นหลัง
        if self.rembg_session:
            try:
                # rembg.remove รับ Input เป็น Numpy Array (BGR/RGB) ได้เลย
                # Output จะเป็น RGBA (มี Alpha Channel โปร่งใส)
                output = remove(img_bgr, session=self.rembg_session)
                
                # Save as PNG (เพื่อเก็บความโปร่งใส)
                cv2.imwrite(full_path, output)
            except Exception as e:
                print(f"⚠️ Rembg Error: {e}, Falling back to OpenCV logic.")
                # ถ้า AI พัง ให้ใช้ OpenCV แบบเดิมสำรอง
                backup = self.processor.remove_background(img_bgr)
                cv2.imwrite(full_path, backup)
        else:
            # ถ้าเครื่องนี้ไม่ได้ลง Rembg ให้ใช้วิธีเดิม
            backup = self.processor.remove_background(img_bgr)
            cv2.imwrite(full_path, backup)

        self.count_saved += 1
        self.btn_record.setText(f"STOP RECORDING ({self.count_saved})")

    def toggle_record(self):
        if self.btn_record.isChecked():
            if not self.entry_class.text().strip():
                QMessageBox.warning(self, "Warning", "Please enter a Class Name!")
                self.btn_record.setChecked(False)
                return
            self.is_recording = True
            self.count_saved = 0
            self.btn_record.setText("STOP RECORDING (0)")
            self.btn_record.setStyleSheet("background-color: #c0392b;") 
            self.entry_class.setEnabled(False)
        else:
            self.is_recording = False
            self.btn_record.setText("START RECORDING")
            self.btn_record.setStyleSheet("background-color: #e74c3c;") 
            self.entry_class.setEnabled(True)
            QMessageBox.information(self, "Done", f"Session saved {self.count_saved} images.")

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.save_dir = folder
            self.lbl_path.setText(f".../{os.path.basename(folder)}")

    def refresh_models(self):
        models = self.detector.get_available_models()
        self.combo_model.clear()
        if models:
            self.combo_model.addItems(models)
            self.load_model(models[0])

    def load_model(self, name):
        if not name: return
        success, msg = self.detector.load_model(os.path.join("models", name))
        print(msg)

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())