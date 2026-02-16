import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QFrame, QSizePolicy, QComboBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

# Core Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.core.config import MODE_PATHS
from src.core.camera import CameraManager
from src.core.detector import ObjectDetector
from src.core.image_utils import ImageProcessor
from src.core.database import DatabaseManager
from src.model_arch.pill_model import PillModel

os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"

class DoctorStation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionForge: Doctor Station (Scanner)")
        self.resize(1200, 800)

        # 1. System
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera = CameraManager()
        self.detector = ObjectDetector() 
        self.processor = ImageProcessor()
        
        self.arcface_transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 2. State
        self.db_manager = None
        self.memory_db = []
        self.current_mode = "pills"
        
        # 3. UI
        self.init_ui()
        self.camera.start(0)
        self.switch_mode("pills")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(30)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0,0,0,0)

        # Header with Mode Switcher
        header = QFrame(); header.setStyleSheet("background-color: #222; padding: 10px;")
        h_lay = QHBoxLayout(header)
        
        lbl_title = QLabel("🩺 VisionForge Scanner")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        h_lay.addWidget(lbl_title)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["💊 PILLS MODE", "📦 BOXES MODE"])
        self.combo_mode.currentIndexChanged.connect(lambda idx: self.switch_mode("pills" if idx==0 else "boxes"))
        self.combo_mode.setFixedWidth(200)
        h_lay.addWidget(self.combo_mode)
        
        layout.addWidget(header)

        # Video
        self.lbl_video = QLabel("Camera")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #000;")
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_video, stretch=4)

        # Result Banner
        self.banner = QFrame()
        self.banner.setFixedHeight(120)
        b_lay = QVBoxLayout(self.banner)
        self.lbl_res = QLabel("READY")
        self.lbl_res.setAlignment(Qt.AlignCenter)
        self.lbl_res.setFont(QFont("Segoe UI", 28, QFont.Bold))
        b_lay.addWidget(self.lbl_res)
        
        layout.addWidget(self.banner, stretch=1)

    def switch_mode(self, mode):
        self.current_mode = mode
        cfg = MODE_PATHS[mode]
        
        # Connect DB & Load Memory
        if self.db_manager: self.db_manager.close()
        self.db_manager = DatabaseManager(cfg["db"])
        
        # 🔥 Preload Vectors for fast search
        self.memory_db = self.db_manager.get_all_vectors()
        print(f"Loaded {len(self.memory_db)} items from {mode}")
        
        # Load Models
        if mode == "pills":
            self.load_arcface(cfg["rec_model"])
        else:
            self.arcface_model = None

    def load_arcface(self, path):
        if not path or not os.path.exists(path): return
        try:
            model = PillModel()
            ckpt = torch.load(path, map_location=self.device)
            clean = {k:v for k,v in ckpt.items() if not k.startswith('head')}
            model.load_state_dict(clean, strict=False)
            model.to(self.device).eval()
            self.arcface_model = model
        except: pass

    def update_logic(self):
        frame = self.camera.get_frame()
        if frame is None: return
        
        processed = self.processor.apply_filters(frame)
        disp = processed.copy()
        
        # Detection
        best_box = self.detector.predict(processed, conf=0.6)
        
        match_name = "UNKNOWN"
        bg_col = "#333"
        
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0 and self.arcface_model:
                vec = self.compute_embedding(crop)
                name, score = self.find_match(vec)
                
                if name:
                    match_name = f"{name} ({score:.0f}%)"
                    bg_col = "#0e4a2e" # Green
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                else:
                    bg_col = "#4a0e0e" # Red
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        self.lbl_res.setText(match_name)
        self.banner.setStyleSheet(f"background-color: {bg_col}; color: white;")
        
        self.show_video(disp)

    def compute_embedding(self, img):
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad(): return self.arcface_model(t).cpu().numpy().flatten()
        except: return None

    def find_match(self, vec):
        if vec is None or not self.memory_db: return None, 0
        best_s, best_n = -1, None
        for item in self.memory_db:
            db_v = item['vector']
            s = np.dot(vec, db_v) / (np.linalg.norm(vec)*np.linalg.norm(db_v))
            if s > best_s: best_s = s; best_n = item['name']
        return (best_n, best_s*100) if best_s > 0.65 else (None, 0)

    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(self.lbl_video.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DoctorStation()
    w.show()
    sys.exit(app.exec())