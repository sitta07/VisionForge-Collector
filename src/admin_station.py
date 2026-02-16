import sys
import os
import time
import cv2
import sqlite3
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# --- 📦 AI & Image Processing Libs ---
from rembg import remove, new_session 
from ultralytics import YOLO 

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QLineEdit, QMessageBox, QSlider, QComboBox, 
                               QSizePolicy, QFileDialog, QGroupBox, QSpinBox, 
                               QProgressBar, QTabWidget, QInputDialog, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QAbstractItemView)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# ลด Log ที่รกหน้าจอ
os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

# --- 🔌 Import Backend Logic ---
try:
    from camera import CameraManager
    from detector import ObjectDetector 
    from image_utils import ImageProcessor
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.models.architecture import PillModel 
except ImportError as e:
    print(f"⚠️ System Warning: Import logic minor issue -> {e}")

# ==========================================
# 🏛️ ENTERPRISE DATABASE MANAGER (SQLite)
# ==========================================
class SqliteDatabaseManager:
    def __init__(self, db_path="data/hospital_main.db"):
        self.db_path = db_path
        self.conn = None
        self.ensure_directory()
        self.connect_db()

    def ensure_directory(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def connect_db(self):
        """เชื่อมต่อ Database และจูน Performance สำหรับ High Scale"""
        if self.conn:
            self.conn.close()
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # 🚀 PERFORMANCE TUNING (สำคัญมากสำหรับ 200M Records)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging (เร็ว + ปลอดภัย)
        self.conn.execute("PRAGMA synchronous=NORMAL") # ลดการรอเขียน Disk เล็กน้อยเพื่อความเร็ว
        
        self.create_tables()

    def create_tables(self):
        # สร้าง Index ที่ชื่อ (name) เพื่อให้ Analytics เร็วระดับสายฟ้าแลบ
        query = """
        CREATE TABLE IF NOT EXISTS drugs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            vector BLOB,          -- เก็บ Vector เป็น Binary (Compact สุดๆ)
            img_path TEXT,
            timestamp REAL
        );
        CREATE INDEX IF NOT EXISTS idx_drug_name ON drugs(name);
        """
        self.conn.executescript(query)
        self.conn.commit()

    def set_db_path(self, new_path):
        self.db_path = new_path
        self.connect_db()

    def create_new_db(self, new_path):
        self.db_path = new_path
        self.connect_db()
        print(f"🆕 Created/Switched to database: {new_path}")

    def add_entry(self, name, vector_np, img_path):
        """เพิ่มข้อมูล (รองรับ Transaction ป้องกันข้อมูลหาย)"""
        try:
            # แปลง Numpy Array -> Bytes (BLOB)
            if vector_np is not None and len(vector_np) > 0:
                vector_blob = vector_np.astype(np.float32).tobytes()
            else:
                vector_blob = None

            with self.conn: # Auto-commit transaction
                self.conn.execute(
                    "INSERT INTO drugs (name, vector, img_path, timestamp) VALUES (?, ?, ?, ?)",
                    (name, vector_blob, img_path, time.time())
                )
            return True
        except Exception as e:
            print(f"❌ DB Insert Error: {e}")
            return False

    def delete_class(self, class_name):
        """ลบ Class ทั้งหมด (เร็วมากเพราะมี Index)"""
        try:
            with self.conn:
                cursor = self.conn.execute("DELETE FROM drugs WHERE name = ?", (class_name,))
                return cursor.rowcount
        except Exception as e:
            print(f"❌ DB Delete Error: {e}")
            return 0

    def get_stats(self):
        """
        🔥 KILLER FEATURE: ใช้ SQL GROUP BY 
        แทนที่จะโหลดข้อมูลมานับใน Python (ซึ่งจะตายถ้ามี 200M rows)
        เราให้ Database นับให้ ผลลัพธ์ได้ในเสี้ยววินาที
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name, COUNT(*) FROM drugs GROUP BY name ORDER BY COUNT(*) DESC")
            return cursor.fetchall() # Returns [(name, count), ...]
        except Exception as e:
            print(f"❌ Stats Error: {e}")
            return []

    def get_all_vectors(self):
        """โหลด Vector ขึ้น RAM สำหรับ Live Search (ถ้าข้อมูลเยอะต้องระวัง RAM หมด)"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name, vector FROM drugs WHERE vector IS NOT NULL")
            rows = cursor.fetchall()
            
            data = []
            for name, blob in rows:
                # แปลง Bytes กลับเป็น Numpy
                vec = np.frombuffer(blob, dtype=np.float32)
                data.append({'name': name, 'vector': vec})
            return data
        except Exception as e:
            print(f"❌ Load Vector Error: {e}")
            return []

# ==========================================
# 🖥️ MAIN UI CLASS
# ==========================================
class AdminStation(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VisionForge: Enterprise Station (200M Scale Ready)")
        self.resize(1600, 950)

        # --- 1. System Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Processing Unit: {self.device}")

        self.recording_active = False
        self.recording_start_time = 0
        self.video_frames_buffer = [] 
        self.last_capture_time = 0    
        
        # Database Init (เปลี่ยนเป็น SQLite DB)
        default_db = os.path.join("data", "hospital_main.db")
        self.db_manager = SqliteDatabaseManager(default_db)
        
        # --- 2. AI Init ---
        self.init_ai_engines()
        self.camera = CameraManager()
        self.detector = ObjectDetector() 
        self.processor = ImageProcessor()
        
        # Variables
        self.current_zoom = 1.0
        self.current_embedding = None 
        self.best_crop_rgba = None
        self.detected_conf = 0.0

        # --- 3. UI Setup ---
        self.apply_styles()
        self.init_ui()

        # --- 4. Start ---
        self.refresh_model_lists() 
        self.start_camera_safe()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(30) 

    def init_ai_engines(self):
        try:
            self.rembg_session = new_session("u2net") 
        except:
            self.rembg_session = None

        self.arcface_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.arcface_model = None

    def start_camera_safe(self):
        for i in range(3):
            try:
                self.camera.start(i)
                if self.camera.cap is not None and self.camera.cap.isOpened(): return
            except: continue

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #f0f0f0; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
            QFrame.Sidebar { background-color: #252526; border-left: 1px solid #3e3e42; }
            QGroupBox { border: 1px solid #454545; border-radius: 6px; margin-top: 12px; font-weight: bold; color: #aaa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #3e3e42; border: 1px solid #555; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background-color: #4e4e52; }
            QPushButton.Primary { background-color: #0e639c; color: white; border: none; font-weight: bold; }
            QPushButton.Danger { background-color: #a4262c; color: white; border: none; font-weight: bold; }
            QLineEdit, QComboBox, QSpinBox { background-color: #333337; border: 1px solid #555; padding: 6px; border-radius: 3px; color: white; }
            QTableWidget { background-color: #252526; gridline-color: #444; border: none; }
            QHeaderView::section { background-color: #333; padding: 4px; border: 1px solid #444; font-weight: bold; }
            QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #4ec9b0; }
        """)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)

        # ================= LEFT: Video =================
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(20, 20, 20, 20)
        
        self.lbl_video = QLabel("Initializing Camera Feed...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #000; border-radius: 8px; border: 2px solid #333;")
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.lbl_video)
        
        stat_bar = QHBoxLayout()
        self.lbl_system_status = QLabel("Ready")
        stat_bar.addWidget(QLabel("Status: ")); stat_bar.addWidget(self.lbl_system_status)
        stat_bar.addStretch()
        btn_zm = QPushButton("-"); btn_zm.setFixedWidth(30); btn_zm.clicked.connect(lambda: self.change_zoom(-1))
        btn_zp = QPushButton("+"); btn_zp.setFixedWidth(30); btn_zp.clicked.connect(lambda: self.change_zoom(1))
        self.lbl_zoom = QLabel("1.0x")
        stat_bar.addWidget(QLabel("Zoom:")); stat_bar.addWidget(btn_zm); stat_bar.addWidget(self.lbl_zoom); stat_bar.addWidget(btn_zp)
        video_layout.addLayout(stat_bar)
        self.main_layout.addWidget(video_container, stretch=3)

        # ================= RIGHT: Sidebar =================
        self.sidebar = QFrame(); self.sidebar.setProperty("class", "Sidebar"); self.sidebar.setFixedWidth(500)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(15, 15, 15, 15)

        # --- 1. DB & CONFIG ---
        grp_config = QGroupBox("🛠️ SYSTEM CONFIGURATION")
        cfg_lay = QVBoxLayout(grp_config)
        
        # DB Selection
        h_db = QHBoxLayout()
        h_db.addWidget(QLabel("DB File:"))
        self.lbl_db_path = QLineEdit(os.path.basename(self.db_manager.db_path))
        self.lbl_db_path.setReadOnly(True)
        btn_load_db = QPushButton("📂"); btn_load_db.setFixedWidth(35); btn_load_db.clicked.connect(self.browse_database)
        btn_new_db = QPushButton("➕"); btn_new_db.setFixedWidth(35); btn_new_db.clicked.connect(self.create_new_database)
        h_db.addWidget(self.lbl_db_path); h_db.addWidget(btn_load_db); h_db.addWidget(btn_new_db)
        cfg_lay.addLayout(h_db)

        # Models
        h1 = QHBoxLayout(); h1.addWidget(QLabel("YOLO:")); 
        self.combo_yolo = QComboBox(); self.combo_yolo.currentTextChanged.connect(self.load_detection_model)
        h1.addWidget(self.combo_yolo, 1); cfg_lay.addLayout(h1)

        h2 = QHBoxLayout(); h2.addWidget(QLabel("ArcFace:"))
        self.combo_arcface = QComboBox(); self.combo_arcface.currentTextChanged.connect(self.load_arcface_model_logic)
        h2.addWidget(self.combo_arcface, 1); cfg_lay.addLayout(h2)

        h3 = QHBoxLayout(); h3.addWidget(QLabel("Conf:"))
        self.sl_conf = QSlider(Qt.Horizontal); self.sl_conf.setRange(10, 95); self.sl_conf.setValue(60)
        self.lbl_conf_val = QLabel("0.60"); self.sl_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v/100:.2f}"))
        h3.addWidget(self.sl_conf, 1); h3.addWidget(self.lbl_conf_val); cfg_lay.addLayout(h3)
        side_layout.addWidget(grp_config)

        # --- 2. TABS ---
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #444; top: -1px; } QTabBar::tab { background: #333; color: #aaa; padding: 8px 20px; margin-right: 2px; } QTabBar::tab:selected { background: #0e639c; color: white; }")
        
        # TAB 1: REGISTRATION
        tab_reg = QWidget(); reg_layout = QVBoxLayout(tab_reg)
        
        # Preview
        prev_group = QGroupBox("LIVE PREVIEW")
        p_lay = QVBoxLayout(prev_group)
        self.lbl_preview = QLabel("Waiting...")
        self.lbl_preview.setFixedSize(220, 180)
        self.lbl_preview.setStyleSheet("background-color: #111; border: 1px dashed #555;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        p_lay.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        reg_layout.addWidget(prev_group)

        # Action
        ops_group = QGroupBox("ACTION")
        o_lay = QVBoxLayout(ops_group)
        o_lay.addWidget(QLabel("Drug Name:"))
        self.input_name = QLineEdit(); self.input_name.setPlaceholderText("Enter Drug Name...")
        self.input_name.setStyleSheet("font-size: 16px; padding: 8px; border: 1px solid #0e639c;")
        o_lay.addWidget(self.input_name)

        self.capture_tabs = QTabWidget()
        # Photo Tab
        t_photo = QWidget(); lp = QVBoxLayout(t_photo)
        self.btn_snap = QPushButton("📸 SNAPSHOT"); self.btn_snap.setProperty("class", "Primary"); self.btn_snap.setFixedHeight(45)
        self.btn_snap.clicked.connect(lambda: self.trigger_save(mode="photo"))
        lp.addWidget(self.btn_snap)
        self.capture_tabs.addTab(t_photo, "Photo")
        
        # Video Tab
        t_video = QWidget(); lv = QVBoxLayout(t_video)
        h_vid = QHBoxLayout(); h_vid.addWidget(QLabel("Sec:")); 
        self.spin_duration = QSpinBox(); self.spin_duration.setRange(1, 60); self.spin_duration.setValue(10)
        h_vid.addWidget(self.spin_duration)
        lv.addLayout(h_vid)
        self.progress_video = QProgressBar(); self.progress_video.setValue(0)
        lv.addWidget(self.progress_video)
        self.btn_rec = QPushButton("🔴 RECORD"); self.btn_rec.setProperty("class", "Primary")
        self.btn_rec.clicked.connect(lambda: self.trigger_save(mode="video"))
        lv.addWidget(self.btn_rec)
        self.capture_tabs.addTab(t_video, "Video")

        o_lay.addWidget(self.capture_tabs)
        reg_layout.addWidget(ops_group)
        reg_layout.addStretch()

        # TAB 2: ANALYTICS
        tab_anl = QWidget(); anl_layout = QVBoxLayout(tab_anl)
        anl_layout.addWidget(QLabel("📊 Database Analysis (Optimized for Large Scale)"))
        
        self.table_anl = QTableWidget()
        self.table_anl.setColumnCount(4)
        self.table_anl.setHorizontalHeaderLabels(["Drug Name", "Count", "Health", "Action"])
        self.table_anl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_anl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_anl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_anl.setSelectionMode(QAbstractItemView.NoSelection)
        anl_layout.addWidget(self.table_anl)

        btn_refresh = QPushButton("🔄 Refresh Data")
        btn_refresh.clicked.connect(self.refresh_analysis_tab)
        anl_layout.addWidget(btn_refresh)

        self.main_tabs.addTab(tab_reg, "📝 Registration")
        self.main_tabs.addTab(tab_anl, "📈 Analytics")
        self.main_tabs.currentChanged.connect(self.on_tab_changed)

        side_layout.addWidget(self.main_tabs)
        self.main_layout.addWidget(self.sidebar)

    # ================= FUNC: Database =================
    def browse_database(self):
        # เปลี่ยน Filter เป็น .db
        fname, _ = QFileDialog.getOpenFileName(self, "Select Database", "data/", "SQLite DB (*.db)")
        if fname:
            self.db_manager.set_db_path(fname)
            self.lbl_db_path.setText(os.path.basename(fname))
            self.refresh_analysis_tab()
            QMessageBox.information(self, "DB Loaded", f"Switched to: {os.path.basename(fname)}")

    def create_new_database(self):
        text, ok = QInputDialog.getText(self, "New Database", "Filename (e.g. hospital_2):")
        if ok and text:
            if not text.endswith(".db"): text += ".db"
            new_path = os.path.join("data", text)
            if os.path.exists(new_path):
                QMessageBox.warning(self, "Error", "File already exists!")
                return
            self.db_manager.create_new_db(new_path)
            self.lbl_db_path.setText(text)
            self.refresh_analysis_tab()

    # ================= FUNC: Analytics =================
    def on_tab_changed(self, index):
        if index == 1: self.refresh_analysis_tab()

    def refresh_analysis_tab(self):
        # 🔥 ดึงข้อมูลเร็วมากเพราะใช้ SQL GROUP BY
        stats = self.db_manager.get_stats()
        self.table_anl.setRowCount(0)
        
        max_count = 50 # Target
        for name, count in stats:
            row = self.table_anl.rowCount()
            self.table_anl.insertRow(row)
            self.table_anl.setItem(row, 0, QTableWidgetItem(name))
            self.table_anl.setItem(row, 1, QTableWidgetItem(str(count)))
            
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            percent = int((count / max_count) * 100)
            pbar.setValue(min(100, percent))
            
            if percent < 30: col = "#a4262c"
            elif percent < 70: col = "#dcdcaa"
            else: col = "#4ec9b0"
            pbar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {col}; }}")
            self.table_anl.setCellWidget(row, 2, pbar)
            
            btn_del = QPushButton("🗑️"); btn_del.setFixedWidth(40)
            btn_del.setStyleSheet("background-color: #552222; border: 1px solid #884444;")
            btn_del.clicked.connect(lambda _, n=name: self.delete_class_logic(n))
            self.table_anl.setCellWidget(row, 3, btn_del)

    def delete_class_logic(self, class_name):
        reply = QMessageBox.question(self, "Confirm Delete", 
                                     f"🚨 ยืนยันการลบ '{class_name}' ?\nการกระทำนี้กู้คืนไม่ได้!",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            deleted = self.db_manager.delete_class(class_name)
            QMessageBox.information(self, "Deleted", f"ลบข้อมูลจำนวน {deleted} รายการเรียบร้อย")
            self.refresh_analysis_tab()

    # ================= FUNC: Logic =================
    def refresh_model_lists(self):
        yolo_models = self.detector.get_available_models()
        self.combo_yolo.addItems(yolo_models)
        if os.path.exists("models"):
            pth_files = [f for f in os.listdir("models") if f.endswith(".pth")]
            self.combo_arcface.addItems(pth_files)
            if "best_model_arcface.pth" in pth_files: self.combo_arcface.setCurrentText("best_model_arcface.pth")

    def load_detection_model(self, name):
        if name: self.detector.load_model(os.path.join("models", name))

    def load_arcface_model_logic(self, name):
        if not name: return
        path = os.path.join("models", name)
        try:
            model = PillModel(num_classes=1000, model_name='convnext_small', embed_dim=512)
            ckpt = torch.load(path, map_location=self.device)
            clean_dict = {k: v for k, v in ckpt.items() if not k.startswith('head')}
            model.load_state_dict(clean_dict, strict=False)
            model.to(self.device).eval()
            self.arcface_model = model
            print(f"✅ ArcFace Loaded: {name}")
        except Exception as e: print(f"❌ ArcFace Load Error: {e}")

    def change_zoom(self, d):
        self.current_zoom = max(1.0, min(5.0, self.current_zoom + d))
        self.lbl_zoom.setText(f"{self.current_zoom:.1f}x")
        self.camera.set_zoom(self.current_zoom)

    # ================= CORE LOOP =================
    def update_logic(self):
        frame = self.camera.get_frame()
        if frame is None: return

        processed = self.processor.apply_filters(frame, zoom=1.0) 
        display = processed.copy()
        
        self.processor.draw_crosshair(display)
        if self.recording_active:
            cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        conf_thresh = self.sl_conf.value() / 100.0
        best_box = self.detector.predict(processed, conf=conf_thresh)
        
        self.best_crop_rgba = None 
        self.current_embedding = None 
        
        match_text = "Scanning..."
        match_color = (200, 200, 200)

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            self.detected_conf = best_box.conf[0].item()
            
            raw_crop = frame[y1:y2, x1:x2]
            if raw_crop.size > 0:
                if not self.recording_active:
                    self.process_and_show_preview(raw_crop)

                if self.arcface_model:
                    self.current_embedding = self.compute_embedding(raw_crop)
                    
                    # 🔥 LIVE SEARCH: โหลดครั้งแรกหรือเมื่อจำเป็น
                    # หมายเหตุ: ถ้าข้อมูล 200M จริงๆ การโหลดขึ้น RAM ตรงนี้จะทำให้คอมค้าง
                    # ใน Production จริงต้องใช้ FAISS Server แยกต่างหาก
                    # แต่สำหรับโค้ดนี้เราจะโหลดเท่าที่มีไปก่อน
                    all_vectors = self.db_manager.get_all_vectors() 
                    found_name, score = self.find_best_match(self.current_embedding, all_vectors)
                    
                    if found_name:
                        match_text = f"{found_name} ({score:.1f}%)"
                        match_color = (0, 255, 0)
                    else:
                        match_text = "New / Unknown"
                        match_color = (0, 165, 255)

                if self.recording_active:
                    self.handle_video_recording(raw_crop)

            color = (0, 255, 0) if not self.recording_active else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
        self.draw_overlay(display, match_text, match_color)
        self.show_video_on_ui(display)
        
        if self.recording_active:
            elapsed = time.time() - self.recording_start_time
            duration = self.spin_duration.value()
            progress = int((elapsed / duration) * 100)
            self.progress_video.setValue(min(100, progress))
            if elapsed >= duration:
                self.stop_recording_and_save()

    def compute_embedding(self, bgr_img):
        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            tensor = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vector = self.arcface_model(tensor).cpu().numpy().flatten()
            return vector
        except: return None

    def find_best_match(self, query_vec, database):
        if query_vec is None or not database: return None, 0.0
        best_score = -1.0; best_name = None
        for item in database:
            db_vec = item['vector']
            if len(db_vec) != len(query_vec): continue
            score = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))
            if score > best_score:
                best_score = score; best_name = item['name']
        return (best_name, best_score * 100) if best_score > 0.65 else (None, best_score * 100)

    def draw_overlay(self, img, text, color):
        h, w = img.shape[:2]
        cv2.putText(img, text, (w - 400, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def process_and_show_preview(self, crop_bgr):
        try:
            if self.rembg_session:
                rgba = remove(crop_bgr, session=self.rembg_session)
            else:
                rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
            self.best_crop_rgba = rgba
            h, w, c = rgba.shape
            qi = QImage(rgba.data, w, h, c*w, QImage.Format_RGBA8888).rgbSwapped()
            self.lbl_preview.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_preview.size(), Qt.KeepAspectRatio))
        except: pass

    # ================= RECORDING =================
    def trigger_save(self, mode):
        # 1. Stop Logic
        if mode == "video" and self.recording_active:
            self.stop_recording_and_save()
            return

        # 2. Validation
        name = self.input_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Alert", "❗ กรุณากรอกชื่อยาก่อน!")
            self.input_name.setFocus(); return

        # 3. Photo
        if mode == "photo":
            if self.best_crop_rgba is None:
                QMessageBox.warning(self, "Alert", "❌ ไม่พบเม็ดยา!")
                return
            self.save_single_entry(self.best_crop_rgba, name, embedding=self.current_embedding)
            QMessageBox.information(self, "Saved", f"✅ บันทึก '{name}' เรียบร้อย!")

        # 4. Video
        elif mode == "video":
            self.recording_active = True
            self.recording_start_time = time.time()
            self.video_frames_buffer = []
            self.last_capture_time = 0
            
            self.btn_rec.setText("⏹️ STOP RECORDING")
            self.btn_rec.setStyleSheet("background-color: #a4262c; color: white;")
            self.capture_tabs.setTabEnabled(0, False)
            self.input_name.setDisabled(True)

    def handle_video_recording(self, crop_bgr):
        if crop_bgr is None: return
        current_t = time.time()
        if current_t - self.last_capture_time > 0.5:
            self.video_frames_buffer.append(crop_bgr.copy())
            self.last_capture_time = current_t

    def stop_recording_and_save(self):
        self.recording_active = False
        name = self.input_name.text().strip()
        
        self.btn_rec.setText("🔴 RECORD VIDEO")
        self.btn_rec.setProperty("class", "Primary"); self.btn_rec.setStyleSheet("")
        self.apply_styles()
        
        self.capture_tabs.setTabEnabled(0, True)
        self.input_name.setDisabled(False)
        self.progress_video.setValue(0)

        count = 0
        if self.video_frames_buffer:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            # Batch Processing for Speed
            for i, raw_frame in enumerate(self.video_frames_buffer):
                if self.rembg_session: rgba = remove(raw_frame, session=self.rembg_session)
                else: rgba = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2BGRA)
                
                vec = self.compute_embedding(raw_frame)
                self.save_single_entry(rgba, name, suffix=f"_vid_{i}", embedding=vec)
                count += 1
            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Success", f"🎥 Video Saved {count} Frames!")
        else:
            QMessageBox.warning(self, "Warning", "❌ ไม่ได้ภาพยาเลย")

    def save_single_entry(self, rgba_img, name, suffix="", embedding=None):
        vector = embedding
        if vector is None and self.arcface_model:
             try:
                rgb = cv2.cvtColor(rgba_img, cv2.COLOR_BGRA2RGB)
                tensor = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
                with torch.no_grad(): vector = self.arcface_model(tensor).cpu().numpy().flatten()
             except: vector = np.array([])
        elif vector is None: vector = np.array([])

        ts = int(time.time() * 1000)
        filename = f"{name}_{ts}{suffix}.png"
        img_path = os.path.join("data/ref_images", filename)
        os.makedirs("data/ref_images", exist_ok=True)
        cv2.imwrite(img_path, rgba_img)

        # Insert into SQLite
        self.db_manager.add_entry(name, vector, img_path)

    def show_video_on_ui(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qi).scaled(1280, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_video.setPixmap(pix)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminStation()
    window.show()
    sys.exit(app.exec())