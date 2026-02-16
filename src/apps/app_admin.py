import sys
import os
import time
import cv2
import numpy as np
import torch
from torchvision import transforms

# --- 📦 AI Libs ---
from rembg import remove, new_session 
from ultralytics import YOLO 

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QLineEdit, QMessageBox, QSlider, QComboBox, 
                               QSizePolicy, QGroupBox, QSpinBox, QProgressBar, 
                               QTabWidget, QTableWidget, QTableWidgetItem, 
                               QHeaderView, QAbstractItemView)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# --- 🔌 Core Logic ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.core.config import MODE_PATHS, BASE_DIR # Import BASE_DIR มาช่วย scan ไฟล์
try:
    from src.core.camera import CameraManager
    from src.core.detector import ObjectDetector
    from src.core.image_utils import ImageProcessor
    from src.core.database import DatabaseManager
    from src.model_arch.pill_model import PillModel 
except ImportError as e:
    print(f"⚠️ Core Import Error: {e}")
    sys.exit(1)

os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

class AdminStation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionForge: Admin Station (Smart Controls)")
        self.resize(1600, 950)

        # 1. Hardware & AI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera = CameraManager()
        self.detector = ObjectDetector() 
        self.processor = ImageProcessor()
        
        # Rembg (โหลดรอไว้เลย แต่จะใช้หรือไม่ขึ้นอยู่กับโหมด)
        try: self.rembg_session = new_session("u2net") 
        except: self.rembg_session = None

        self.arcface_transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.arcface_model = None

        # 2. State
        self.db_manager = None
        self.current_mode = "pills"
        self.current_config = {}
        
        self.recording_active = False
        self.video_frames_buffer = [] 
        self.best_crop_rgba = None
        self.current_embedding = None

        # 3. UI & Start
        self.apply_styles()
        self.init_ui()
        self.camera.start(0)
        
        # Init Default
        self.switch_mode("pills")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(30) 

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0)

        # --- LEFT: VIDEO ---
        self.video_container = QWidget()
        v_layout = QVBoxLayout(self.video_container)
        self.lbl_video = QLabel("Initializing...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #000;")
        v_layout.addWidget(self.lbl_video)
        self.main_layout.addWidget(self.video_container, stretch=3)

        # --- RIGHT: SIDEBAR ---
        self.sidebar = QFrame(); self.sidebar.setFixedWidth(480)
        self.sidebar.setStyleSheet("background-color: #252526;")
        self.side_layout = QVBoxLayout(self.sidebar)

        # 1. System Control (Dropdowns & Sliders)
        grp_sys = QGroupBox("⚙️ SYSTEM CONTROL")
        l_sys = QVBoxLayout(grp_sys)
        l_sys.setSpacing(10)
        
        # Mode
        l_sys.addWidget(QLabel("Operation Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["💊 PILLS MODE", "📦 BOXES MODE"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_change)
        l_sys.addWidget(self.combo_mode)
        
        l_sys.addWidget(self.create_line())

        # Dropdowns (แทนปุ่ม Browse)
        l_sys.addWidget(QLabel("Database File:"))
        self.combo_db = QComboBox()
        self.combo_db.currentTextChanged.connect(self.on_db_change)
        l_sys.addWidget(self.combo_db)

        l_sys.addWidget(QLabel("YOLO Model:"))
        self.combo_yolo = QComboBox()
        self.combo_yolo.currentTextChanged.connect(self.on_yolo_change)
        l_sys.addWidget(self.combo_yolo)

        l_sys.addWidget(QLabel("ArcFace Model:"))
        self.combo_arcface = QComboBox()
        self.combo_arcface.currentTextChanged.connect(self.on_arcface_change)
        l_sys.addWidget(self.combo_arcface)

        l_sys.addWidget(self.create_line())

        # YOLO Confidence Slider
        h_conf = QHBoxLayout()
        h_conf.addWidget(QLabel("Detection Conf:"))
        self.lbl_conf_val = QLabel("0.60")
        h_conf.addWidget(self.lbl_conf_val)
        l_sys.addLayout(h_conf)
        
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(10, 95) # 0.10 - 0.95
        self.sl_conf.setValue(60)
        self.sl_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v/100:.2f}"))
        l_sys.addWidget(self.sl_conf)

        self.side_layout.addWidget(grp_sys)

        # 2. Registration Tab
        self.tabs = QTabWidget()
        t_reg = QWidget(); l_reg = QVBoxLayout(t_reg)
        
        self.lbl_preview = QLabel("Preview")
        self.lbl_preview.setFixedSize(220, 180)
        self.lbl_preview.setStyleSheet("border: 1px dashed #555; background: #111;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        l_reg.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        
        l_reg.addWidget(QLabel("Drug Name / ID:"))
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Scan or type name...")
        self.input_name.textChanged.connect(self.check_drug_status)
        l_reg.addWidget(self.input_name)
        
        self.lbl_name_status = QLabel("")
        self.lbl_name_status.setStyleSheet("font-size: 12px; font-weight: bold;")
        l_reg.addWidget(self.lbl_name_status)
        
        h_btn = QHBoxLayout()
        self.btn_snap = QPushButton("📸 SNAP"); self.btn_snap.setFixedHeight(45)
        self.btn_snap.clicked.connect(lambda: self.trigger_save("photo"))
        self.btn_rec = QPushButton("🔴 RECORD VIDEO"); self.btn_rec.setFixedHeight(45)
        self.btn_rec.clicked.connect(lambda: self.trigger_save("video"))
        h_btn.addWidget(self.btn_snap); h_btn.addWidget(self.btn_rec)
        l_reg.addLayout(h_btn)
        
        self.progress_video = QProgressBar(); self.progress_video.setValue(0)
        l_reg.addWidget(self.progress_video)
        l_reg.addStretch()

        # 3. Analytics Tab
        t_anl = QWidget(); l_anl = QVBoxLayout(t_anl)
        self.table_anl = QTableWidget(0, 3)
        self.table_anl.setHorizontalHeaderLabels(["Name", "Count", "Action"])
        self.table_anl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        l_anl.addWidget(self.table_anl)
        btn_ref = QPushButton("🔄 Refresh"); btn_ref.clicked.connect(self.refresh_analytics)
        l_anl.addWidget(btn_ref)

        self.tabs.addTab(t_reg, "📝 Register")
        self.tabs.addTab(t_anl, "📊 Analytics")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.side_layout.addWidget(self.tabs)
        self.main_layout.addWidget(self.sidebar)

    # --- UI Helpers ---
    def create_line(self):
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #444;")
        return line

    def populate_dropdown(self, combo, folder, ext_list, default_file=None):
        """Scan folder และใส่ชื่อไฟล์ลง Dropdown"""
        combo.blockSignals(True) # หยุด event ชั่วคราวกันลั่น
        combo.clear()
        
        if not os.path.exists(folder):
            combo.addItem("Folder not found")
            combo.blockSignals(False)
            return

        files = [f for f in os.listdir(folder) if f.endswith(tuple(ext_list))]
        if not files:
            combo.addItem("No files found")
        else:
            combo.addItems(files)
            # พยายามเลือกไฟล์ default ถ้ามี
            if default_file:
                default_name = os.path.basename(default_file)
                index = combo.findText(default_name)
                if index >= 0: combo.setCurrentIndex(index)
        
        combo.blockSignals(False)

    # ================= LOGIC: SWITCH MODE & LOAD =================
    def on_mode_change(self, index):
        mode = "pills" if index == 0 else "boxes"
        self.switch_mode(mode)

    def switch_mode(self, mode):
        self.current_mode = mode
        self.current_config = MODE_PATHS[mode] # ดึงค่าจาก config.py
        print(f"🔄 Switching to: {mode.upper()}")

        # 1. Update Dropdowns (Scan Files)
        # DB อยู่ใน data/{mode}/
        db_folder = os.path.dirname(self.current_config["db"])
        self.populate_dropdown(self.combo_db, db_folder, [".db"], self.current_config["db"])
        
        # Model Weights อยู่ใน model_weights/ (ใช้ร่วมกัน แต่ default ต่างกัน)
        weights_folder = os.path.join(BASE_DIR, "model_weights") # หรือ models ตามที่คุณตั้ง
        # ถ้าหา folder ไม่เจอ ให้ลองถอยกลับไปหา models
        if not os.path.exists(weights_folder): weights_folder = os.path.join(BASE_DIR, "models")

        self.populate_dropdown(self.combo_yolo, weights_folder, [".pt", ".onnx"], self.current_config["yolo_model"])
        self.populate_dropdown(self.combo_arcface, weights_folder, [".pth"], self.current_config["rec_model"])

        # 2. Trigger Load Logic (ผ่าน Event ของ Dropdown หรือเรียกตรงๆ)
        # เนื่องจากเรา blockSignals ไว้ตอน populate เราต้องเรียกโหลดเองครั้งแรก
        self.on_db_change(self.combo_db.currentText())
        self.on_yolo_change(self.combo_yolo.currentText())
        self.on_arcface_change(self.combo_arcface.currentText())
        
        self.input_name.clear()
        self.lbl_name_status.setText("")

    # --- Dropdown Handlers ---
    def on_db_change(self, fname):
        if not fname or "not found" in fname: return
        # Construct full path
        folder = os.path.dirname(self.current_config["db"])
        full_path = os.path.join(folder, fname)
        
        if self.db_manager: self.db_manager.close()
        self.db_manager = DatabaseManager(full_path)
        self.refresh_analytics()
        print(f"💾 DB Connected: {fname}")

    def on_yolo_change(self, fname):
        if not fname or "not found" in fname: return
        # หา path จริง (ลองหาใน model_weights หรือ models)
        folder = os.path.join(BASE_DIR, "model_weights")
        if not os.path.exists(folder): folder = os.path.join(BASE_DIR, "models")
        
        full_path = os.path.join(folder, fname)
        self.detector.load_model(full_path)
        print(f"👁️ YOLO Loaded: {fname}")

    def on_arcface_change(self, fname):
        if self.current_mode == "boxes": 
            self.arcface_model = None
            return # กล่องไม่ใช้ ArcFace

        if not fname or "not found" in fname: return
        folder = os.path.join(BASE_DIR, "model_weights")
        if not os.path.exists(folder): folder = os.path.join(BASE_DIR, "models")
        
        full_path = os.path.join(folder, fname)
        self.load_arcface_model(full_path)

    def load_arcface_model(self, path):
        try:
            model = PillModel(num_classes=1000, model_name='convnext_small', embed_dim=512)
            ckpt = torch.load(path, map_location=self.device)
            clean_dict = {k: v for k, v in ckpt.items() if not k.startswith('head')}
            model.load_state_dict(clean_dict, strict=False)
            model.to(self.device).eval()
            self.arcface_model = model
            print(f"🧠 ArcFace Ready: {os.path.basename(path)}")
        except Exception as e:
            print(f"❌ ArcFace Failed: {e}")

    # ================= LOGIC: MAIN LOOP =================
    def update_logic(self):
        frame = self.camera.get_frame()
        if frame is None: return
        
        processed = self.processor.apply_filters(frame)
        display = processed.copy()
        self.processor.draw_crosshair(display)

        # 🔥 Use Slider Confidence
        conf = self.sl_conf.value() / 100.0
        best_box = self.detector.predict(processed, conf=conf)
        
        self.best_crop_rgba = None
        
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            raw_crop = frame[y1:y2, x1:x2]
            
            if raw_crop.size > 0:
                # 🔥 Logic Background Removal (สำคัญ!)
                # ยาเม็ด -> ตัด (เพราะเทรนมาแบบตัด)
                # กล่อง -> ไม่ตัด (เพื่อรักษาขอบ)
                use_rembg = self.current_config.get("use_rembg", True)
                
                if use_rembg and self.rembg_session:
                    try: rgba = remove(raw_crop, session=self.rembg_session)
                    except: rgba = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2BGRA)
                else:
                    rgba = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2BGRA) # No Rembg for Boxes
                
                self.best_crop_rgba = rgba
                
                # Show Preview (เฉพาะตอนไม่อัด)
                if not self.recording_active:
                    self.show_preview(rgba)
                
                # Live Embedding (เฉพาะ Pills)
                if self.arcface_model:
                    self.current_embedding = self.compute_embedding(raw_crop)
                
                if self.recording_active:
                    self.video_frames_buffer.append(raw_crop)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.show_video(display)
        
        if self.recording_active and len(self.video_frames_buffer) > 200: 
            self.trigger_save("video")

    # ... (ส่วน Save, Analytics, Check Drug ยังเหมือนเดิม Code เก่าดีอยู่แล้ว) ...
    # (Copy Logic เดิมของ check_drug_status, trigger_save, refresh_analytics มาใส่ตรงนี้ได้เลย)
    
    def check_drug_status(self):
        name = self.input_name.text().strip()
        if not name or not self.db_manager:
            self.lbl_name_status.setText("")
            return
        try:
            cursor = self.db_manager.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM drugs WHERE name = ?", (name,))
            count = cursor.fetchone()[0]
            if count > 0:
                self.lbl_name_status.setText(f"⚠️ Existing: มีแล้ว {count} รูป (เพิ่มได้)")
                self.lbl_name_status.setStyleSheet("color: #ffaa00;")
            else:
                self.lbl_name_status.setText("✅ New Drug")
                self.lbl_name_status.setStyleSheet("color: #00ff00;")
        except: pass

    def trigger_save(self, mode):
        name = self.input_name.text().strip()
        if not name: 
            QMessageBox.warning(self, "Error", "Please enter drug name")
            return

        if mode == "video":
            if self.recording_active:
                self.recording_active = False
                self.btn_rec.setText("🔴 RECORD VIDEO")
                self.btn_rec.setStyleSheet("")
                self.process_video_buffer(name)
            else:
                self.recording_active = True
                self.btn_rec.setText("⏹️ STOP")
                self.btn_rec.setStyleSheet("background-color: #c0392b;")
                self.video_frames_buffer = []
        
        elif mode == "photo":
            if self.best_crop_rgba is not None:
                self.save_entry(self.best_crop_rgba, name, "_snap", self.current_embedding)
                QMessageBox.information(self, "Saved", "Snapshot Saved!")
                self.refresh_analytics()
                self.check_drug_status()

    def process_video_buffer(self, name):
        count = 0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        use_rembg = self.current_config.get("use_rembg", True)
        
        for i, frame in enumerate(self.video_frames_buffer):
            if i % 3 == 0:
                if use_rembg and self.rembg_session:
                    try: rgba = remove(frame, session=self.rembg_session)
                    except: rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                else:
                    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    
                vec = self.compute_embedding(frame) if self.arcface_model else None
                self.save_entry(rgba, name, f"_vid_{i}", vec)
                count += 1
                
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Saved", f"Saved {count} frames")
        self.video_frames_buffer = []
        self.refresh_analytics()
        self.check_drug_status()

    def save_entry(self, rgba_img, name, suffix, vector=None):
        ts = int(time.time() * 1000)
        filename = f"{name}_{ts}{suffix}.png"
        save_dir = self.current_config["ref_img_dir"]
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        full_path = os.path.join(save_dir, filename)
        cv2.imwrite(full_path, rgba_img)
        
        if vector is None and self.arcface_model:
             if rgba_img.shape[2] == 4: img_bgr = cv2.cvtColor(rgba_img, cv2.COLOR_BGRA2BGR)
             else: img_bgr = rgba_img
             vector = self.compute_embedding(img_bgr)
        self.db_manager.add_entry(name, vector, full_path)

    def compute_embedding(self, bgr_img):
        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            t = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad(): return self.arcface_model(t).cpu().numpy().flatten()
        except: return None

    def refresh_analytics(self):
        if not self.db_manager: return
        stats = self.db_manager.get_stats()
        self.table_anl.setRowCount(0)
        for name, count in stats:
            row = self.table_anl.rowCount()
            self.table_anl.insertRow(row)
            self.table_anl.setItem(row, 0, QTableWidgetItem(name))
            self.table_anl.setItem(row, 1, QTableWidgetItem(str(count)))
            btn_del = QPushButton("🗑️"); btn_del.clicked.connect(lambda _, n=name: self.delete_logic(n))
            self.table_anl.setCellWidget(row, 2, btn_del)

    def delete_logic(self, name):
        if QMessageBox.question(self, "Delete", f"Delete '{name}'?") == QMessageBox.Yes:
            self.db_manager.delete_class(name)
            self.refresh_analytics()

    def on_tab_changed(self, idx):
        if idx == 1: self.refresh_analytics()

    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))
    
    def show_preview(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGBA8888).rgbSwapped()
        self.lbl_preview.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio))
        
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #f0f0f0; font-family: 'Segoe UI'; font-size: 14px; }
            QLineEdit, QComboBox { background-color: #333; padding: 5px; border: 1px solid #555; }
            QPushButton { background-color: #444; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #555; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AdminStation()
    w.show()
    sys.exit(app.exec())