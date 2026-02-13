import sys
import os
import json
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from rembg import remove, new_session
from ultralytics import YOLO

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFrame,
    QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

try:
    from camera import CameraManager
    from image_utils import ImageProcessor
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.models.architecture import PillModel
except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    sys.exit(1)


# =========================================================
# YOLO DETECTOR (SAFE VERSION รองรับ .pt / .onnx)
# =========================================================
class ObjectDetector:
    def __init__(self):
        self.model = None

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self.model = YOLO(path)
        print(f"✅ YOLO Loaded: {path}")

    def predict(self, frame, conf=0.6):
        if self.model is None:
            return None

        results = self.model(frame, conf=conf)

        if not results or len(results[0].boxes) == 0:
            return None

        return results[0].boxes[0]


# =========================================================
# MAIN ADMIN STATION
# =========================================================
class AdminStation(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VisionForge: Admin Station Pro")
        self.resize(1400, 900)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===============================
        # Rembg
        # ===============================
        try:
            self.rembg_session = new_session("u2net")
            print("✅ Rembg Engine Ready!")
        except:
            self.rembg_session = None

        # ===============================
        # ArcFace
        # ===============================
        self.arcface_model = self.load_arcface_model("models/best_model_arcface.pth")

        # ===============================
        # Components
        # ===============================
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()

        self.best_crop_rgba = None
        self.db_path = os.path.join("data", "hospital_drug_db.json")
        self.ensure_database()

        self.apply_styles()
        self.init_ui()

        # 🔥 โหลด YOLO ตรง ๆ (ไม่ใช้ refresh_models)
        self.load_yolo_model()

        self.start_camera_safe()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(30)

    # =====================================================
    # LOAD YOLO
    # =====================================================
    def load_yolo_model(self):
        model_path = "models/best.onnx"  # เปลี่ยนเป็น .pt ได้

        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"YOLO model not found:\n{model_path}")
            sys.exit(1)

        try:
            self.detector.load_model(model_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"YOLO Load Failed:\n{e}")
            sys.exit(1)

    # =====================================================
    # ARC FACE
    # =====================================================
    def load_arcface_model(self, path):
        try:
            model = PillModel(
                num_classes=1000,
                model_name='convnext_small',
                embed_dim=512
            )

            if os.path.exists(path):
                ckpt = torch.load(path, map_location=self.device)
                clean_dict = {k: v for k, v in ckpt.items()
                              if not k.startswith('head')}
                model.load_state_dict(clean_dict, strict=False)

            model.to(self.device).eval()
            print("✅ ArcFace Ready")
            return model
        except Exception as e:
            print(f"❌ ArcFace Load Fail: {e}")
            return None

    # =====================================================
    # CAMERA
    # =====================================================
    def start_camera_safe(self):
        for i in range(3):
            try:
                self.camera.start(i)
                if self.camera.cap and self.camera.cap.isOpened():
                    print(f"✅ Camera Opened Index {i}")
                    return
            except:
                continue

        QMessageBox.critical(self, "Error", "No Camera Found")
        sys.exit(1)

    # =====================================================
    # DATABASE
    # =====================================================
    def ensure_database(self):
        os.makedirs("data/ref_images", exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump([], f)

    # =====================================================
    # UI
    # =====================================================
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI'; }
        """)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QHBoxLayout(self.central_widget)

        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: black;")
        layout.addWidget(self.lbl_video, stretch=3)

        side = QVBoxLayout()

        self.lbl_preview = QLabel("Preview")
        self.lbl_preview.setFixedSize(250, 200)
        self.lbl_preview.setStyleSheet("background:black;")
        side.addWidget(self.lbl_preview)

        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Enter Drug Name...")
        side.addWidget(self.input_name)

        self.lbl_duplicate = QLabel("")
        side.addWidget(self.lbl_duplicate)

        self.btn_save = QPushButton("💾 SAVE")
        self.btn_save.clicked.connect(self.save_logic)
        side.addWidget(self.btn_save)

        layout.addLayout(side)

    # =====================================================
    # MAIN LOOP
    # =====================================================
    def update_logic(self):
        frame = self.camera.get_frame()
        if frame is None:
            return

        processed = self.processor.apply_filters(frame, zoom=1.0)
        display = processed.copy()

        best_box = self.detector.predict(processed, conf=0.6)

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            cv2.rectangle(display, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            raw_crop = frame[y1:y2, x1:x2]

            if raw_crop.size > 0:
                if self.rembg_session:
                    rgba_crop = remove(raw_crop, session=self.rembg_session)
                else:
                    rgba_crop = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2BGRA)

                self.best_crop_rgba = rgba_crop
                self.show_preview(rgba_crop)
                self.check_similarity_live(rgba_crop)

        self.show_video(display)

    # =====================================================
    # SIMILARITY
    # =====================================================
    def check_similarity_live(self, rgba_crop):
        if not os.path.exists(self.db_path):
            return

        with open(self.db_path, 'r') as f:
            db = json.load(f)

        if not db:
            self.lbl_duplicate.setText("🆕 ยังไม่เคยลงทะเบียน")
            return

        rgb_crop = cv2.cvtColor(rgba_crop, cv2.COLOR_BGRA2RGB)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            input_vec = self.arcface_model(input_tensor)

        best_score = -1
        best_name = None

        for entry in db:
            db_vec = torch.tensor(entry["vector"]).to(self.device)
            score = F.cosine_similarity(
                input_vec,
                db_vec.unsqueeze(0)
            ).item()

            if score > best_score:
                best_score = score
                best_name = entry["name"]

        if best_score > 0.80:
            self.lbl_duplicate.setText(
                f"⚠️ Duplicate: {best_name} ({best_score:.4f})"
            )
        else:
            self.lbl_duplicate.setText(
                f"🆕 New Drug ({best_score:.4f})"
            )

    # =====================================================
    # SAVE
    # =====================================================
    def save_logic(self):
        name = self.input_name.text().strip()

        if not name or self.best_crop_rgba is None:
            QMessageBox.warning(self, "Warning",
                                "กรุณาใส่ชื่อและวางยาในตำแหน่ง")
            return

        rgb_crop = cv2.cvtColor(self.best_crop_rgba, cv2.COLOR_BGRA2RGB)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vector = self.arcface_model(input_tensor)\
                .cpu().numpy().flatten().tolist()

        ts = int(time.time())
        img_path = f"data/ref_images/{name}_{ts}.png"
        cv2.imwrite(img_path, self.best_crop_rgba)

        with open(self.db_path, 'r+') as f:
            db = json.load(f)
            db.append({
                "name": name,
                "vector": vector,
                "img": img_path
            })
            f.seek(0)
            json.dump(db, f, indent=4)

        QMessageBox.information(self, "Saved", f"Saved '{name}'")
        self.input_name.clear()

    # =====================================================
    # DISPLAY
    # =====================================================
    def show_video(self, img):
        h, w, c = img.shape
        qi = QImage(img.data, w, h, c*w,
                    QImage.Format_RGB888).rgbSwapped()

        self.lbl_video.setPixmap(
            QPixmap.fromImage(qi).scaled(
                self.lbl_video.size(),
                Qt.KeepAspectRatio
            )
        )

    def show_preview(self, img):
        h, w, c = img.shape
        fmt = QImage.Format_RGBA8888 if c == 4 else QImage.Format_RGB888

        qi = QImage(img.data, w, h, c*w, fmt).rgbSwapped()

        self.lbl_preview.setPixmap(
            QPixmap.fromImage(qi).scaled(
                250, 200,
                Qt.KeepAspectRatio
            )
        )


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminStation()
    window.show()
    sys.exit(app.exec())
