import sys
import os
import json
import time
import cv2
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
                               QLineEdit, QMessageBox, QSlider, QComboBox, QSizePolicy)
from PySide6.QtCore import Qt, QTimer
# 🔥 แก้บั๊กบรรทัดนี้: เพิ่ม QImage เข้ามาแล้ว
from PySide6.QtGui import QImage, QPixmap 

# --- 🔌 Import Backend Logic ---
try:
    from camera import CameraManager
    from detector import ObjectDetector 
    from image_utils import ImageProcessor
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.models.architecture import PillModel 
except ImportError as e:
    print(f"⚠️ Error: {e}")
    sys.exit(1)

class AdminStation(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VisionForge: Admin Station Pro (Hybrid Engine)")
        self.resize(1400, 900)

        # --- 1. System Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initial AI Rembg
        try:
            self.rembg_session = new_session("u2net") 
            print("✅ Rembg Engine Ready!")
        except:
            self.rembg_session = None

        # Initial ArcFace Model (Senior Mode: Ignore Head mismatch)
        self.arcface_model = self.load_arcface_model("models/best_model_arcface.pth")
        
        self.camera = CameraManager()
        self.detector = ObjectDetector() 
        self.processor = ImageProcessor()
        
        # Variables
        self.current_zoom = 1.0
        self.best_crop_rgba = None
        self.db_path = os.path.join("data", "hospital_drug_db.json")
        self.ensure_database()

        # --- 2. UI Setup ---
        self.apply_styles()
        self.init_ui()

        # --- 3. Start Loop ---
        self.refresh_models() 
        self.start_camera_safe()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(30)

    def load_arcface_model(self, path):
        try:
            model = PillModel(num_classes=1000, model_name='convnext_small', embed_dim=512)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=self.device)
                # กรอง head ออกเพื่อให้โหลด weights ได้แม้จำนวน class จะไม่ตรงกัน
                clean_dict = {k: v for k, v in ckpt.items() if not k.startswith('head')}
                model.load_state_dict(clean_dict, strict=False)
            model.to(self.device).eval()
            return model
        except Exception as e:
            print(f"❌ ArcFace Load Fail: {e}")
            return None

    def start_camera_safe(self):
        for i in range(3):
            try:
                self.camera.start(i)
                if self.camera.cap is not None and self.camera.cap.isOpened(): return
            except: continue

    def ensure_database(self):
        os.makedirs("data/ref_images", exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w', encoding='utf-8') as f: json.dump([], f)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI'; }
            QFrame#Sidebar { background-color: #1e1e1e; border-left: 1px solid #333; }
            QFrame.Card { background-color: #252526; border-radius: 8px; border: 1px solid #333; padding: 12px; }
            QLabel.CardTitle { color: #3b8ed0; font-weight: bold; font-size: 13px; }
            QPushButton { background-color: #3a3a3a; border-radius: 4px; padding: 8px; font-weight: bold; }
            QPushButton#SaveBtn { background-color: #27ae60; font-size: 16px; height: 50px; }
            QLineEdit, QComboBox { background-color: #1e1e1e; border: 1px solid #444; padding: 8px; color: white; }
        """)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0)

        # LEFT: Video
        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: black;")
        self.main_layout.addWidget(self.lbl_video, stretch=3)

        # RIGHT: Sidebar
        self.sidebar = QFrame(); self.sidebar.setObjectName("Sidebar"); self.sidebar.setFixedWidth(400)
        self.side_layout = QVBoxLayout(self.sidebar)

        # 1. System Control
        sys_card = QFrame(); sys_card.setProperty("class", "Card")
        sys_lay = QVBoxLayout(sys_card)
        sys_lay.addWidget(QLabel("SYSTEM CONTROL", objectName="CardTitle"))
        
        sys_lay.addWidget(QLabel("YOLO Model (.pt/.onnx):"))
        self.combo_model = QComboBox()
        self.combo_model.currentTextChanged.connect(self.load_detection_model)
        sys_lay.addWidget(self.combo_model)

        zoom_lay = QHBoxLayout()
        zoom_lay.addWidget(QLabel("Zoom:"))
        btn_m = QPushButton("-"); btn_m.clicked.connect(lambda: self.change_zoom(-1))
        btn_p = QPushButton("+"); btn_p.clicked.connect(lambda: self.change_zoom(1))
        self.lbl_zoom = QLabel("1.0x")
        zoom_lay.addStretch(); zoom_lay.addWidget(btn_m); zoom_lay.addWidget(self.lbl_zoom); zoom_lay.addWidget(btn_p)
        sys_lay.addLayout(zoom_lay)

        sys_lay.addWidget(QLabel("Confidence Threshold:"))
        self.sl_conf = QSlider(Qt.Horizontal); self.sl_conf.setRange(10, 95); self.sl_conf.setValue(60)
        self.lbl_conf = QLabel("0.60")
        conf_lay = QHBoxLayout(); conf_lay.addWidget(self.sl_conf); conf_lay.addWidget(self.lbl_conf)
        self.sl_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"{v/100:.2f}"))
        sys_lay.addLayout(conf_lay)
        self.side_layout.addWidget(sys_card)

        # 2. Preview (Rembg)
        prev_card = QFrame(); prev_card.setProperty("class", "Card")
        prev_lay = QVBoxLayout(prev_card)
        prev_lay.addWidget(QLabel("CROP PREVIEW (AI REMBG)", objectName="CardTitle"))
        self.lbl_preview = QLabel("Waiting...")
        self.lbl_preview.setFixedSize(250, 200)
        self.lbl_preview.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        prev_lay.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        self.side_layout.addWidget(prev_card)

        # 3. Form
        info_card = QFrame(); info_card.setProperty("class", "Card")
        info_lay = QVBoxLayout(info_card)
        info_lay.addWidget(QLabel("DRUG REGISTRATION", objectName="CardTitle"))
        self.input_name = QLineEdit(); self.input_name.setPlaceholderText("Enter Drug Name...")
        info_lay.addWidget(self.input_name)
        self.lbl_status = QLabel("Status: Searching...")
        info_lay.addWidget(self.lbl_status)
        self.side_layout.addWidget(info_card)

        self.side_layout.addStretch()
        self.btn_save = QPushButton("💾 SAVE TO DATABASE"); self.btn_save.setObjectName("SaveBtn")
        self.btn_save.clicked.connect(self.save_logic)
        self.side_layout.addWidget(self.btn_save)

        self.main_layout.addWidget(self.sidebar)

    def refresh_models(self):
        models = self.detector.get_available_models()
        self.combo_model.addItems(models)

    def load_detection_model(self, name):
        if name: self.detector.load_model(os.path.join("models", name))

    def change_zoom(self, d):
        self.current_zoom = max(1.0, min(5.0, self.current_zoom + d))
        self.lbl_zoom.setText(f"{self.current_zoom:.1f}x")
        self.camera.set_zoom(self.current_zoom)

    # ==========================================
    # 🔥 CORE LOOP: YOLO + BBox + Rembg
    # ==========================================
    def update_logic(self):
        frame = self.camera.get_frame()
        if frame is None: return

        processed = self.processor.apply_filters(frame, zoom=1.0)
        display = processed.copy()
        self.processor.draw_crosshair(display)

        # 1. Detection (Ultralytics)
        conf_val = self.sl_conf.value() / 100.0
        best_box = self.detector.predict(processed, conf=conf_val)
        
        if best_box is not None:
            # ดึงพิกัดจาก YOLO Result
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            conf_score = best_box.conf[0].item()
            
            # 🔥 วาดเส้นกรอบ (Bounding Box)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Pill {conf_score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 2. Crop & Rembg ลบพื้นหลัง
            raw_crop = frame[y1:y2, x1:x2]
            if raw_crop.size > 0:
                if self.rembg_session:
                    rgba_crop = remove(raw_crop, session=self.rembg_session)
                    self.best_crop_rgba = rgba_crop
                    self.show_preview(rgba_crop)
                else:
                    self.best_crop_rgba = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2BGRA)
                    self.show_preview(raw_crop)
                self.lbl_status.setText("✅ Pill Locked!")
        else:
            self.best_crop_rgba = None
            self.lbl_status.setText("🔍 Searching...")

        self.show_video(display)

    def save_logic(self):
        name = self.input_name.text().strip()
        if not name or self.best_crop_rgba is None:
            QMessageBox.warning(self, "Warning", "กรุณาระบุชื่อยาและวางยาในตำแหน่ง!")
            return

        # ArcFace Embedding
        rgb_crop = cv2.cvtColor(self.best_crop_rgba, cv2.COLOR_BGRA2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(rgb_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vector = self.arcface_model(input_tensor).cpu().numpy().flatten().tolist()

        ts = int(time.time())
        img_path = f"data/ref_images/{name}_{ts}.png"
        cv2.imwrite(img_path, self.best_crop_rgba)

        # Write to JSON
        with open(self.db_path, 'r+') as f:
            db = json.load(f); db.append({"name": name, "vector": vector, "img": img_path})
            f.seek(0); json.dump(db, f, indent=4)
        
        QMessageBox.information(self, "Saved", f"บันทึกข้อมูล '{name}' เรียบร้อย!")
        self.input_name.clear()

    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))

    def show_preview(self, img):
        # 🛡️ แก้ไขเรียบร้อย: QImage จะไม่พ่น NameError อีกต่อไป
        h, w, c = img.shape
        fmt = QImage.Format_RGBA8888 if c==4 else QImage.Format_RGB888
        qi = QImage(img.data, w, h, c*w, fmt).rgbSwapped()
        self.lbl_preview.setPixmap(QPixmap.fromImage(qi).scaled(250, 200, Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminStation()
    window.show()
    sys.exit(app.exec())