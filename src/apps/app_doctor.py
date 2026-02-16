import sys
import os
import time
import cv2
import numpy as np
import torch
from torchvision import transforms

# --- 📦 AI Libs (No Rembg) ---
from ultralytics import YOLO 

# --- 🖥️ GUI Libs ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QMessageBox, QSlider, QGroupBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# --- 🔌 Core Logic ---
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.core.config import MODE_PATHS, BASE_DIR

# Handle Imports safely
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
        self.setWindowTitle("VisionForge: Inference Station (Pure Crop Mode)")
        self.resize(1400, 850)

        # 1. Hardware & AI Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Processing Device: {self.device}")
        
        self.camera = CameraManager()
        self.detector = ObjectDetector() 
        self.processor = ImageProcessor()
        
        # --- ❌ REMOVED REMBG SESSION ---
        # No background removal init here. Pure Speed.

        # ArcFace Transform
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
        self.current_embedding = None
        self.best_crop_rgba = None

        # 3. UI & Start
        self.apply_styles()
        self.init_ui()
        self.camera.start(0)
        
        # Init Default Mode
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
        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #000; border-right: 2px solid #333;")
        v_layout.addWidget(self.lbl_video)
        self.main_layout.addWidget(self.video_container, stretch=3)

        # --- RIGHT: SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(400)
        self.sidebar.setStyleSheet("background-color: #252526;")
        self.side_layout = QVBoxLayout(self.sidebar)
        self.side_layout.setSpacing(20)

        # 1. Mode Selection (Buttons)
        grp_mode = QGroupBox("🎮 OPERATION MODE")
        l_mode = QVBoxLayout(grp_mode)
        
        h_btns = QHBoxLayout()
        self.btn_pills = QPushButton("💊 PILLS")
        self.btn_pills.setCheckable(True)
        self.btn_pills.setFixedHeight(50)
        self.btn_pills.clicked.connect(lambda: self.switch_mode("pills"))
        
        self.btn_boxes = QPushButton("📦 BOXES")
        self.btn_boxes.setCheckable(True)
        self.btn_boxes.setFixedHeight(50)
        self.btn_boxes.clicked.connect(lambda: self.switch_mode("boxes"))
        
        h_btns.addWidget(self.btn_pills)
        h_btns.addWidget(self.btn_boxes)
        l_mode.addLayout(h_btns)
        self.side_layout.addWidget(grp_mode)

        # 2. Settings (Slider Only)
        grp_set = QGroupBox("⚙️ SETTINGS")
        l_set = QVBoxLayout(grp_set)
        
        h_conf = QHBoxLayout()
        h_conf.addWidget(QLabel("Detection Conf:"))
        self.lbl_conf_val = QLabel("0.60")
        self.lbl_conf_val.setStyleSheet("color: #00ff00; font-weight: bold;")
        h_conf.addWidget(self.lbl_conf_val)
        l_set.addLayout(h_conf)
        
        self.sl_conf = QSlider(Qt.Horizontal)
        self.sl_conf.setRange(10, 95)
        self.sl_conf.setValue(60) # Default 0.6
        self.sl_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v/100:.2f}"))
        l_set.addWidget(self.sl_conf)
        self.side_layout.addWidget(grp_set)

        # 3. Status / Preview Crop
        grp_stat = QGroupBox("👁️ LIVE CROP (RAW)")
        l_stat = QVBoxLayout(grp_stat)
        self.lbl_preview = QLabel("Waiting...")
        self.lbl_preview.setFixedSize(220, 220)
        self.lbl_preview.setStyleSheet("border: 1px dashed #555; background: #111;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        l_stat.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        
        self.lbl_status = QLabel("System Ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #aaa; font-size: 12px;")
        l_stat.addWidget(self.lbl_status)
        
        self.side_layout.addWidget(grp_stat)
        
        # Spacer
        self.side_layout.addStretch()
        
        # Version
        lbl_ver = QLabel("VisionForge AI - Pure Crop v1.0")
        lbl_ver.setAlignment(Qt.AlignCenter)
        lbl_ver.setStyleSheet("color: #555;")
        self.side_layout.addWidget(lbl_ver)

        self.main_layout.addWidget(self.sidebar)

    # ================= LOGIC: AUTO SWITCH & LOAD =================
    def switch_mode(self, mode):
        self.current_mode = mode
        
        # Update UI Buttons
        if mode == "pills":
            self.btn_pills.setChecked(True)
            self.btn_boxes.setChecked(False)
            self.btn_pills.setStyleSheet("background-color: #007acc; color: white;")
            self.btn_boxes.setStyleSheet("")
        else:
            self.btn_pills.setChecked(False)
            self.btn_boxes.setChecked(True)
            self.btn_boxes.setStyleSheet("background-color: #d35400; color: white;")
            self.btn_pills.setStyleSheet("")

        print(f"🔄 Switching to: {mode.upper()}")
        self.lbl_status.setText(f"Loading {mode.upper()} resources...")
        QApplication.processEvents() # Force UI Update

        # --- 1. Auto Load Database ---
        self.load_auto_database(mode)

        # --- 2. Auto Load YOLO (best.onnx) ---
        self.load_auto_yolo()

        # --- 3. Auto Load ArcFace (.pth) ---
        # 🔥🔥 FIX HERE: โหลด Model ทั้ง 2 โหมด ไม่ปิดกั้น Boxes แล้ว 🔥🔥
        self.load_auto_arcface()
        
        self.lbl_status.setText(f"✅ Mode: {mode.upper()} Active")

    def load_auto_database(self, mode):
        """Finds .db in data/{mode} automatically"""
        db_folder = os.path.join(BASE_DIR, "data", mode)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder, exist_ok=True)

        # Find first .db file
        db_files = [f for f in os.listdir(db_folder) if f.endswith(".db")]
        
        if db_files:
            target_db = os.path.join(db_folder, db_files[0])
            print(f"📂 Found DB: {target_db}")
        else:
            target_db = os.path.join(db_folder, "default.db")
            print(f"⚠️ No DB found, creating default: {target_db}")

        # Connect
        if self.db_manager:
            try: self.db_manager.close()
            except: pass
        
        try:
            self.db_manager = DatabaseManager(target_db)
            print(f"✅ DB Connected: {os.path.basename(target_db)}")
        except Exception as e:
            print(f"❌ DB Error: {e}")
            self.db_manager = None

    def load_auto_yolo(self):
        """Finds best.onnx"""
        possible_paths = [
            os.path.join(BASE_DIR, "models", "best.onnx"),
            os.path.join(BASE_DIR, "model_weights", "best.onnx"),
            "best.onnx"
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                self.detector.load_model(path)
                print(f"👁️ YOLO Loaded: {path}")
                found = True
                break
        
        if not found:
            print("❌ Error: 'best.onnx' not found")
            self.lbl_status.setText("❌ Error: YOLO model missing")

    def load_auto_arcface(self):
        """Finds any .pth file"""
        folder = os.path.join(BASE_DIR, "models")
        if not os.path.exists(folder):
            folder = os.path.join(BASE_DIR, "model_weights")
        
        if os.path.exists(folder):
            pth_files = [f for f in os.listdir(folder) if f.endswith(".pth")]
            if pth_files:
                target_pth = os.path.join(folder, pth_files[0])
                try:
                    model = PillModel(num_classes=1000, model_name='convnext_small', embed_dim=512)
                    ckpt = torch.load(target_pth, map_location=self.device)
                    clean_dict = {k: v for k, v in ckpt.items() if not k.startswith('head')}
                    model.load_state_dict(clean_dict, strict=False)
                    model.to(self.device).eval()
                    self.arcface_model = model
                    print(f"🧠 ArcFace Loaded: {target_pth}")
                except Exception as e:
                    print(f"❌ ArcFace Error: {e}")
            else:
                 print("⚠️ No .pth file found for ArcFace")

    # ================= LOGIC: MAIN LOOP =================
    def update_logic(self):
            frame = self.camera.get_frame()
            if frame is None: return
            
            # --- ❌ จุดที่ทำให้เพี้ยน (เดิม) ---
            # processed = self.processor.apply_filters(frame)
            # display = processed.copy()
            # best_box = self.detector.predict(processed, conf=conf) 
            
            # --- ✅ วิธีแก้ (ใหม่) ---
            # ใช้ frame ต้นฉบับทำทุกอย่าง เพื่อให้ Coordinate ตรงกัน 100%
            display = frame.copy() 
            
            # ถ้าอยากใส่ Filter ให้ใส่ที่ display เพื่อการแสดงผล
            # แต่ตอน predict ให้ใช้ frame ดิบ หรือ display ที่ขนาดยังเท่า frame
            # (ถ้า apply_filters ของคุณมีการย่อภาพ ห้ามใช้ภาพนั้นมา predict คู่กับการ crop frame)
            
            h, w = display.shape[:2]
            conf = self.sl_conf.value() / 100.0
            
            # 🔥 CHANGE 1: Predict จาก frame โดยตรง (เพื่อให้ได้พิกัดตรงกับภาพจริง)
            best_box = self.detector.predict(frame, conf=conf)
            
            identified_name = "WAITING..."
            status_color = (100, 100, 100)

            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # --- 🔥 PADDING Logic ---
                pad = 15 
                y1_p = max(0, y1 - pad)
                x1_p = max(0, x1 - pad)
                y2_p = min(h, y2 + pad)
                x2_p = min(w, x2 + pad)
                
                # 🔥 CHANGE 2: ตัดจาก frame (ซึ่งตอนนี้พิกัด x1,y1 ตรงกันแล้ว)
                raw_crop = frame[y1_p:y2_p, x1_p:x2_p]
                
                if raw_crop.size > 0:
                    # แปลงเป็น BGRA
                    rgba = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2BGRA)
                    
                    self.best_crop_rgba = rgba
                    self.show_preview(rgba)
                    
                    # --- RECOGNITION LOGIC ---
                    if self.arcface_model is None:
                        identified_name = "⚠️ NO MODEL"
                        status_color = (0, 0, 255)
                    elif self.db_manager is None:
                        identified_name = "⚠️ NO DB"
                        status_color = (0, 0, 255)
                    else:
                        self.current_embedding = self.compute_embedding(raw_crop)
                        
                        if self.current_embedding is not None:
                            name, score = self.db_manager.search(self.current_embedding)
                            if name:
                                identified_name = f"{name.upper()} ({score:.2f})"
                                status_color = (0, 255, 0)
                            else:
                                identified_name = "UNKNOWN"
                                status_color = (0, 0, 255)
                        else:
                            identified_name = "EMBED ERR"

                # วาดลงบน display (ซึ่งขนาดเท่า frame)
                cv2.rectangle(display, (x1, y1), (x2, y2), status_color, 2)

            # --- DRAW OVERLAY ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            margin = 20

            (text_w, text_h), _ = cv2.getTextSize(identified_name, font, font_scale, thickness)
            text_x = w - text_w - margin
            text_y = h - margin

            cv2.rectangle(display, 
                        (text_x - 10, text_y - text_h - 10), 
                        (w - margin + 5, h - margin + 5), 
                        (0, 0, 0), -1)
            
            cv2.putText(display, identified_name, (text_x, text_y), 
                        font, font_scale, status_color, thickness, cv2.LINE_AA)

            self.show_video(display)
    def compute_embedding(self, bgr_img):
        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            t = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad(): 
                return self.arcface_model(t).cpu().numpy().flatten()
        except: 
            return None

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
            QGroupBox { border: 1px solid #444; margin-top: 10px; padding-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QPushButton { background-color: #444; border-radius: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = AdminStation()
    w.show()
    sys.exit(app.exec())