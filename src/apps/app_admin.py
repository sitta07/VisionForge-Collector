import sys
import os
import time
import cv2
import numpy as np
import torch
import subprocess
from torchvision import transforms

# --- 📦 AI Libs ---
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

# Try Import Logic
try:
    from src.core.config import MODE_PATHS, BASE_DIR
    from src.core.camera import CameraManager
    from src.core.detector import ObjectDetector
    from src.core.image_utils import ImageProcessor
    from src.core.database import DatabaseManager
    from src.model_arch.pill_model import PillModel 
except ImportError as e:
    print(f"⚠️ Core Import Error: {e}")
    # Mock classes for standalone testing if needed
    BASE_DIR = os.getcwd()
    MODE_PATHS = {
        "pills": {"raw_dir": "./data/pills", "yolo_model": "yolov8n.pt", "db": "pills.db"},
        "boxes": {"raw_dir": "./data/boxes", "yolo_model": "yolov8n.pt", "db": "boxes.db"}
    }
    class CameraManager:
        def start(self, id): pass
        def get_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8)
    class ObjectDetector:
        def load_model(self, p): pass
        def predict(self, i, conf): return None
    class ImageProcessor:
        def draw_crosshair(self, i): pass
    class DatabaseManager:
        def __init__(self, p): self.conn = None
        def close(self): pass
        def search(self, v): return None, 0.0
        def get_stats(self): return []
        def add_entry(self, n, v, p): pass
        def delete_class(self, n): pass
    class PillModel:
        def __init__(self, **k): pass
        def load_state_dict(self, d, strict): pass
        def to(self, d): return self
        def eval(self): pass
        def __call__(self, x): return torch.randn(1, 512) # Mock output

os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

class AdminStation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionForge: Admin Station (Auto Zoom & Model Edition)")
        self.resize(1600, 950)

        # 1. Hardware & AI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera = CameraManager()
        self.detector = ObjectDetector() 
        self.processor = ImageProcessor()
        
        # ArcFace Transforms
        self.arcface_transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        
        # Init Default (Will trigger auto zoom & model load)
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
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(480)
        self.sidebar.setStyleSheet("background-color: #252526;")
        self.side_layout = QVBoxLayout(self.sidebar)

        # 1. System Control
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

        # Database
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

        # --- YOLO Confidence Slider ---
        h_conf = QHBoxLayout()
        h_conf.addWidget(QLabel("Detection Conf:"))
        self.lbl_conf_val = QLabel("0.60")
        h_conf.addWidget(self.lbl_conf_val)
        l_sys.addLayout(h_conf)
        
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(10, 95)
        self.sl_conf.setValue(65)
        self.sl_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v/100:.2f}"))
        l_sys.addWidget(self.sl_conf)

        # --- [NEW] Recognition Accuracy Slider ---
        h_rec = QHBoxLayout()
        h_rec.addWidget(QLabel("Rec. Accuracy:"))
        self.lbl_rec_val = QLabel("0.60") # Default Label
        h_rec.addWidget(self.lbl_rec_val)
        l_sys.addLayout(h_rec)

        self.sl_rec = QSlider(Qt.Horizontal)
        self.sl_rec.setRange(10, 99) # Range 0.10 - 0.99
        self.sl_rec.setValue(60)     # Default Value 0.60
        self.sl_rec.valueChanged.connect(lambda v: self.lbl_rec_val.setText(f"{v/100:.2f}"))
        l_sys.addWidget(self.sl_rec)
        # -----------------------------------------

        self.side_layout.addWidget(grp_sys)

        # 2. Registration Tab
        self.tabs = QTabWidget()
        t_reg = QWidget()
        l_reg = QVBoxLayout(t_reg)
        
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
        self.btn_snap = QPushButton("📸 SNAP")
        self.btn_snap.setFixedHeight(45)
        self.btn_snap.clicked.connect(lambda: self.trigger_save("photo"))
        self.btn_rec = QPushButton("🔴 RECORD VIDEO")
        self.btn_rec.setFixedHeight(45)
        self.btn_rec.clicked.connect(lambda: self.trigger_save("video"))
        h_btn.addWidget(self.btn_snap)
        h_btn.addWidget(self.btn_rec)
        l_reg.addLayout(h_btn)
        
        self.progress_video = QProgressBar()
        self.progress_video.setValue(0)
        l_reg.addWidget(self.progress_video)
        l_reg.addStretch()

        # 3. Analytics Tab
        t_anl = QWidget()
        l_anl = QVBoxLayout(t_anl)
        self.table_anl = QTableWidget(0, 3)
        self.table_anl.setHorizontalHeaderLabels(["Name", "Count", "Action"])
        self.table_anl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        l_anl.addWidget(self.table_anl)
        btn_ref = QPushButton("🔄 Refresh")
        btn_ref.clicked.connect(self.refresh_analytics)
        l_anl.addWidget(btn_ref)

        self.tabs.addTab(t_reg, "📝 Register")
        self.tabs.addTab(t_anl, "📊 Analytics")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.side_layout.addWidget(self.tabs)
        self.main_layout.addWidget(self.sidebar)

    # --- UI Helpers ---
    def create_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #444;")
        return line

    def populate_dropdown(self, combo, folder, ext_list, default_file=None):
        combo.blockSignals(True)
        combo.clear()
        
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            combo.addItem("(empty folder)")
            combo.blockSignals(False)
            return

        files = [f for f in os.listdir(folder) if f.endswith(tuple(ext_list))]
        if not files:
            combo.addItem("(no files)")
        else:
            combo.addItems(files)
            if default_file:
                default_name = os.path.basename(default_file)
                index = combo.findText(default_name)
                if index >= 0: 
                    combo.setCurrentIndex(index)
        
        combo.blockSignals(False)

    # ================= LOGIC: SWITCH MODE & LOAD =================
    def on_mode_change(self, index):
        mode = "pills" if index == 0 else "boxes"
        self.switch_mode(mode)

    def switch_mode(self, mode):
        self.current_mode = mode
        self.current_config = MODE_PATHS[mode]
        print(f"🔄 Switching to: {mode.upper()}")

        # 1. AUTO ZOOM LOGIC
        if mode == "pills":
            print("🔍 Auto Zoom: Setting to 60 (Pills)")
            self.set_hardware_zoom(60)
        elif mode == "boxes":
            print("🔍 Auto Zoom: Setting to 50 (Boxes)")
            self.set_hardware_zoom(50)

        # 2. Database
        db_folder = os.path.join(BASE_DIR, "data", mode)
        db_name = self.current_config.get("db", f"{mode}.db") 
        default_db_name = os.path.basename(db_name)
        
        if not os.path.exists(db_folder):
            os.makedirs(db_folder, exist_ok=True)
        
        self.populate_dropdown(self.combo_db, db_folder, [".db"], default_db_name)
        
        # Create Default DB if empty
        if self.combo_db.currentText() in ["(empty folder)", "(no files)"]:
            default_db_path = os.path.join(db_folder, default_db_name)
            if self.db_manager:
                try: self.db_manager.close()
                except: pass
            self.db_manager = DatabaseManager(default_db_path)
            self.combo_db.blockSignals(True)
            self.combo_db.clear()
            self.combo_db.addItem(default_db_name)
            self.combo_db.blockSignals(False)
        
        # 3. YOLO
        weights_folder = os.path.join(BASE_DIR, "model_weights")
        if not os.path.exists(weights_folder): 
            weights_folder = os.path.join(BASE_DIR, "models")
        
        default_yolo = os.path.basename(self.current_config.get("yolo_model", "best.onnx"))
        self.populate_dropdown(self.combo_yolo, weights_folder, [".pt", ".onnx"], default_yolo)
        
        # 4. [UPDATED] Auto ArcFace Selection
        if mode == "pills":
            target_model = "best_pills.pth" # ใช้โมเดล pills
        else:
            target_model = "best_boxes.pth" # ใช้โมเดล boxes
            
        print(f"🎯 Target ArcFace: {target_model}")
        self.populate_dropdown(self.combo_arcface, weights_folder, [".pth"], target_model)

        # Force load the target model directly
        full_model_path = os.path.join(weights_folder, target_model)
        if os.path.exists(full_model_path):
            self.load_arcface_model(full_model_path)
        else:
            print(f"⚠️ Warning: Auto-load model '{target_model}' not found in {weights_folder}")

        # 5. Trigger Other Loads
        self.on_db_change(self.combo_db.currentText())
        self.on_yolo_change(self.combo_yolo.currentText())
        # Note: We already loaded arcface manually above, so we don't need to trigger on_arcface_change here
        
        self.input_name.clear()
        self.lbl_name_status.setText("")

    # --- Hardware Zoom Helper ---
    def set_hardware_zoom(self, value):
        cmd = ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl", f"zoom_absolute={value}"]
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"⚠️ Zoom Warning: Could not set hardware zoom ({e})")

    # --- Dropdown Handlers ---
    def on_db_change(self, fname):
        if not fname or "empty" in fname or "no files" in fname: 
            self.db_manager = None
            return
        
        try:
            db_folder = os.path.join(BASE_DIR, "data", self.current_mode)
            full_path = os.path.join(db_folder, fname)
            
            if not os.path.exists(db_folder):
                os.makedirs(db_folder, exist_ok=True)
            
            if self.db_manager: 
                try: self.db_manager.close()
                except: pass
            
            self.db_manager = DatabaseManager(full_path)
            self.refresh_analytics()
            print(f"💾 DB Connected ({self.current_mode.upper()}): {fname}")
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.db_manager = None

    def on_yolo_change(self, fname):
        if not fname or "no files" in fname: return
        folder = os.path.join(BASE_DIR, "model_weights")
        if not os.path.exists(folder): folder = os.path.join(BASE_DIR, "models")
        full_path = os.path.join(folder, fname)
        if os.path.exists(full_path):
            self.detector.load_model(full_path)
            print(f"👁️ YOLO Loaded: {fname}")

    def on_arcface_change(self, fname):
        if not fname or "no files" in fname or "not used" in fname: return
        
        folder = os.path.join(BASE_DIR, "model_weights")
        if not os.path.exists(folder): folder = os.path.join(BASE_DIR, "models")
        
        full_path = os.path.join(folder, fname)
        if os.path.exists(full_path):
            self.load_arcface_model(full_path)

    def load_arcface_model(self, path):
        try:
            model = PillModel(
    num_classes=1000,  # เลขอะไรก็ได้ เพราะ Head ไม่ได้ใช้
    model_name='convnext_small', 
    embed_dim=512,     # ต้องตรงกับตอนเทรน
    use_cbam=False     # ปิดเพื่อให้โหลด Weight เก่าได้
)
            ckpt = torch.load(path, map_location=self.device)
            
            # --- 🔧 จุดแก้: ปรับ Key ให้ตรงกัน (Handle DataParallel save) ---
            clean_dict = {}
            for k, v in ckpt.items():
                # ตัด head ออกตามเดิม
                if k.startswith('head'): continue
                
                # แก้ชื่อ key ถ้ามี module. นำหน้า (มักเกิดตอนเทรนด้วยหลาย GPU)
                new_k = k.replace("module.", "") 
                clean_dict[new_k] = v
            # -----------------------------------------------------------

            # โหลดและเช็คว่า Key หายไปเยอะไหม
            missing, unexpected = model.load_state_dict(clean_dict, strict=False)
            
            print(f"🧠 Loading Model: {os.path.basename(path)}")
            if len(missing) > 0:
                print(f"⚠️ Warning: Missing Keys ({len(missing)}): {missing[:5]} ...")
                # ถ้า Missing เยอะแสดงว่าโมเดลเป็น Random -> ต้องแก้ชื่อ Key ให้ตรง
            else:
                print("✅ Model Weights Loaded Perfectly!")

            model.to(self.device).eval()
            self.arcface_model = model
            
        except Exception as e:
            print(f"❌ ArcFace Failed: {e}")

    # ================= LOGIC: MAIN LOOP =================
    def update_logic(self):
            # 1. รับ Frame
            frame = self.camera.get_frame()
            if frame is None: return
            
            display = frame.copy()
            self.processor.draw_crosshair(display)

            h, w = display.shape[:2]

            # 3. ตรวจจับ
            conf = self.sl_conf.value() / 100.0
            
            best_box = self.detector.predict(frame, conf=conf)
            
            self.best_crop_rgba = None
            identified_name = "SEARCHING..."
            status_color = (200, 200, 200)

            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                pad = 15
                y1_p = max(0, y1 - pad)
                x1_p = max(0, x1 - pad)
                y2_p = min(h, y2 + pad)
                x2_p = min(w, x2 + pad)

                raw_crop = frame[y1_p:y2_p, x1_p:x2_p]
                
                if raw_crop.size > 0:
                    rgba = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2BGRA)
                    self.best_crop_rgba = rgba
                    
                    if not self.recording_active:
                        self.show_preview(rgba)
                    
                    # --- AI Recognition Logic ---
                    if self.arcface_model:
                        self.current_embedding = self.compute_embedding(raw_crop)
                        
                        if self.current_embedding is not None and self.db_manager:
                            name, score = self.db_manager.search(self.current_embedding)
                            
                            # [UPDATED] Check against Slider Value
                            rec_threshold = self.sl_rec.value() / 100.0
                            
                            if name and score >= rec_threshold:
                                identified_name = f"{name.upper()} ({score:.2f})"
                                status_color = (0, 255, 0)
                                cv2.putText(display, f"{name} ({score:.2f})", (x1, y2 + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                identified_name = "UNKNOWN"
                                status_color = (0, 0, 255)
                                # Show score even if unknown for debugging
                                display_score = f"{score:.2f}" if score else "0.00"
                                cv2.putText(display, f"Unknown ({display_score})", (x1, y2 + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                    if self.recording_active:
                        self.video_frames_buffer.append(raw_crop)

                cv2.rectangle(display, (x1, y1), (x2, y2), status_color, 2)
                cv2.putText(display, f"Conf: {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4. Draw Overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            margin = 30
            (text_w, text_h), baseline = cv2.getTextSize(identified_name, font, font_scale, thickness)
            text_x = w - text_w - margin
            text_y = h - margin
            cv2.rectangle(display, (text_x - 15, text_y - text_h - 15), 
                        (w - margin + 10, h - margin + 10), (0, 0, 0), -1)
            cv2.putText(display, identified_name, (text_x, text_y), 
                        font, font_scale, status_color, thickness, cv2.LINE_AA)

            self.show_video(display)
            
            if self.recording_active and len(self.video_frames_buffer) > 200: 
                self.trigger_save("video")

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
                self.lbl_name_status.setText("✅ New Object")
                self.lbl_name_status.setStyleSheet("color: #00ff00;")
        except: 
            pass

    def trigger_save(self, mode):
        name = self.input_name.text().strip()
        if not name: 
            QMessageBox.warning(self, "Error", "Please enter name")
            return
        
        if not self.db_manager:
            QMessageBox.critical(self, "Database Error", "❌ Database not connected!")
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
            else:
                QMessageBox.warning(self, "No Detection", "No object detected!")

    def process_video_buffer(self, name):
        count = 0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        for i, frame in enumerate(self.video_frames_buffer):
            if i % 3 == 0:
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
        
        save_dir = self.current_config.get("ref_img_dir", 
                                           os.path.join(BASE_DIR, "data", self.current_mode, "ref_images"))
        
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        
        full_path = os.path.join(save_dir, filename)
        cv2.imwrite(full_path, rgba_img)
        
        if vector is None and self.arcface_model:
            # Convert back to BGR for embedding calculation if needed
            if rgba_img.shape[2] == 4: 
                img_bgr = cv2.cvtColor(rgba_img, cv2.COLOR_BGRA2BGR)
            else: 
                img_bgr = rgba_img
            vector = self.compute_embedding(img_bgr)
        
        self.db_manager.add_entry(name, vector, full_path)
        print(f"✅ Saved to {self.current_mode} DB: {name}")

    def compute_embedding(self, bgr_img):
            try:
                rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                t = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
                
                with torch.no_grad(): 
                    # --- แก้ตรงนี้: เพิ่ม F.normalize ---
                    emb = self.arcface_model(t)
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1) # <--- สำคัญมาก!
                    return emb.cpu().numpy().flatten()
                    # ---------------------------------
            except Exception as e: 
                print(f"Embedding Error: {e}")
                return None

    def refresh_analytics(self):
        if not self.db_manager: 
            return
        stats = self.db_manager.get_stats()
        self.table_anl.setRowCount(0)
        for name, count in stats:
            row = self.table_anl.rowCount()
            self.table_anl.insertRow(row)
            self.table_anl.setItem(row, 0, QTableWidgetItem(name))
            self.table_anl.setItem(row, 1, QTableWidgetItem(str(count)))
            btn_del = QPushButton("🗑️")
            btn_del.clicked.connect(lambda _, n=name: self.delete_logic(n))
            self.table_anl.setCellWidget(row, 2, btn_del)

    def delete_logic(self, name):
        if QMessageBox.question(self, "Delete", f"Delete '{name}'?") == QMessageBox.Yes:
            self.db_manager.delete_class(name)
            self.refresh_analytics()

    def on_tab_changed(self, idx):
        if idx == 1: 
            self.refresh_analytics()

    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(
            self.lbl_video.size(), Qt.KeepAspectRatio))
    
    def show_preview(self, img):
        h, w, c = img.shape
        fmt = QImage.Format_RGBA8888 if c == 4 else QImage.Format_RGB888
        qi = QImage(img.data, w, h, c*w, fmt).rgbSwapped()
        self.lbl_preview.setPixmap(QPixmap.fromImage(qi).scaled(
            self.lbl_preview.size(), Qt.KeepAspectRatio))
        
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