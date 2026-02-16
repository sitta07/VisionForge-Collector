"""
AI SMART TECH - Modern Vision System
Card-Based UI with Smooth Interactions + FIXED FLICKERING BOXES
Version: 4.1 - Anti-Flicker Update
"""

import sys
import os
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Optional, Dict, List
import logging
from collections import defaultdict

# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSlider, QGroupBox, QScrollArea,
    QGridLayout
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QImage, QPixmap, QColor, QPalette, QFont, QPainter, QPen

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.core.config import MODE_PATHS, BASE_DIR

try:
    from src.core.camera import CameraManager
    from src.core.detector import ObjectDetector
    from src.core.image_utils import ImageProcessor
    from src.core.database import DatabaseManager
    from src.model_arch.pill_model import PillModel
except ImportError as e:
    print(f"⚠️ Core Import Error: {e}")
    sys.exit(1)


class DetectionTracker:
    """ป้องกันการกระพริบด้วย temporal smoothing"""
    
    def __init__(self, memory_frames: int = 5, min_stable_frames: int = 1):
        self.memory_frames = memory_frames
        self.min_stable_frames = min_stable_frames
        self.detections = {}  # name -> {count: int, frames_seen: int, last_seen_time: float, last_bbox: tuple}
        self.current_frame_time = time.time()
    
    def update(self, detected_items: Dict[str, int], detected_bboxes: Dict[str, tuple]):
        """อัพเดต detection history และกรอง noise"""
        self.current_frame_time = time.time()
        
        # อัพเดต items ที่เจอในเฟรมนี้
        for name, count in detected_items.items():
            if name not in self.detections:
                self.detections[name] = {
                    'count': count,
                    'frames_seen': 1,
                    'last_seen_time': self.current_frame_time,
                    'last_bbox': detected_bboxes.get(name)
                }
            else:
                self.detections[name]['count'] = count
                self.detections[name]['frames_seen'] = min(self.detections[name]['frames_seen'] + 1, 10)  # cap at 10
                self.detections[name]['last_seen_time'] = self.current_frame_time
                if name in detected_bboxes:
                    self.detections[name]['last_bbox'] = detected_bboxes[name]
        
        # ลด frames_seen ของ items ที่ไม่เจอในเฟรมนี้
        names_to_remove = []
        for name in self.detections:
            if name not in detected_items:
                self.detections[name]['frames_seen'] -= 1
                
                # ลบถ้าไม่เห็นนานเกินไป
                if self.detections[name]['frames_seen'] <= -self.memory_frames:
                    names_to_remove.append(name)
        
        for name in names_to_remove:
            del self.detections[name]
    
    def get_stable_detections(self) -> Dict[str, int]:
        """คืนค่าเฉพาะ detections ที่มั่นคง (เห็นมากพอ)"""
        stable = {}
        for name, data in self.detections.items():
            # แสดงเฉพาะ items ที่เห็นมากกว่า min_stable_frames หรือเคยเห็นมั่นคงมาก่อน
            if data['frames_seen'] >= self.min_stable_frames:
                stable[name] = max(0, data['count'])
        return stable
    
    def get_bbox(self, name: str) -> Optional[tuple]:
        """คืนค่า bbox ล่าสุดของ item"""
        if name in self.detections:
            return self.detections[name]['last_bbox']
        return None


class DetectionCard(QFrame):
    """Modern card widget - ปรับให้เล็กลงเพื่อลง Grid 3 คอลัมน์"""
    
    def __init__(self, name: str, count: int, image: Optional[QPixmap] = None):
        super().__init__()
        self.item_name = name
        self.item_count = count
        self.is_selected = False
        
        # ปรับขนาดใหม่ให้ลงตัวกับ 3 คอลัมน์ (125x130)
        self.setFixedSize(125, 130) 
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5) 
        layout.setSpacing(3)
        
        # Image container - เตี้ยลงเพื่อประหยัดพื้นที่
        self.image_container = QFrame()
        self.image_container.setFixedSize(115, 65) 
        self.image_container.setStyleSheet("background-color: #f8f9fa; border-radius: 6px;")
        
        img_layout = QVBoxLayout(self.image_container)
        img_layout.setContentsMargins(2, 2, 2, 2)
        
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        
        if image:
            # Scale ภาพให้เล็กลงตาม container
            scaled = image.scaled(110, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_image.setPixmap(scaled)
        else:
            self.lbl_image.setText("📦")
            self.lbl_image.setStyleSheet("font-size: 24px;")
        
        img_layout.addWidget(self.lbl_image)
        layout.addWidget(self.image_container)
        
        # Name - กระชับตัวอักษร
        self.lbl_name = QLabel(name.upper())
        self.lbl_name.setAlignment(Qt.AlignCenter)
        self.lbl_name.setStyleSheet("font-size: 10px; font-weight: bold; color: #444;")
        layout.addWidget(self.lbl_name)
        
        # Count Badge - เล็กลงแต่เด่น
        self.lbl_count = QLabel(f"×{count}")
        self.lbl_count.setAlignment(Qt.AlignCenter)
        self.lbl_count.setFixedHeight(20)
        self.lbl_count.setStyleSheet("""
            background-color: #E3F2FD;
            color: #1976D2;
            font-size: 11px;
            font-weight: 800;
            border-radius: 10px;
        """)
        layout.addWidget(self.lbl_count)
        
        self.update_style()
    
    def set_image(self, pixmap: QPixmap):
        """Update card image"""
        scaled = pixmap.scaled(100, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(scaled)
    
    def set_selected(self, selected: bool):
        """Toggle selection state"""
        self.is_selected = selected
        self.update_style()
    
    def toggle_selection(self):
        """Toggle selection on/off"""
        self.is_selected = not self.is_selected
        self.update_style()
        return self.is_selected
    
    def update_style(self):
        """Update card appearance based on state"""
        if self.is_selected:
            self.setStyleSheet("""
                DetectionCard {
                    background-color: white;
                    border: 3px solid #1976D2;
                    border-radius: 10px;
                }
            """)
            self.lbl_count.setStyleSheet("""
                background-color: #1976D2;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 12px;
                padding: 3px 10px;
            """)
        else:
            self.setStyleSheet("""
                DetectionCard {
                    background-color: white;
                    border: 2px solid #e0e0e0;
                    border-radius: 10px;
                }
                DetectionCard:hover {
                    border: 2px solid #bbb;
                }
            """)
            self.lbl_count.setStyleSheet("""
                background-color: #E3F2FD;
                color: #1976D2;
                font-size: 13px;
                font-weight: bold;
                border-radius: 12px;
                padding: 3px 10px;
            """)
    
    def mousePressEvent(self, event):
        """Handle click event"""
        self.clicked()
    
    def clicked(self):
        """Emit click signal (to be connected)"""
        pass


class ModernVisionStation(QMainWindow):
    """Modern card-based vision system with anti-flicker"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI SMART TECH | Vision System")
        self.resize(1600, 900)
        
        # Hardware & AI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()
        
        self.arcface_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.arcface_model = None
        
        # State
        self.db_manager = None
        self.current_mode = "pills"
        self.selected_item_name = None
        self.last_detection_time = time.time()
        self.detection_interval = 0.15
        
        # Card tracking
        self.detection_cards = {}  # name -> DetectionCard
        self.detection_images = {}  # name -> best crop image
        
        # *** ANTI-FLICKER: Detection tracker ***
        self.detection_tracker = DetectionTracker(memory_frames=5, min_stable_frames=1)
        
        # UI Setup
        self.setup_modern_ui()
        
        # Start
        self.camera.start(0)
        self.switch_mode("pills")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def setup_modern_ui(self):
        """Setup modern card-based interface"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === LEFT: VIDEO PANEL ===
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #f8f9fa;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)
        
        # Header with gradient
        header = QFrame()
        header.setFixedHeight(100)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 16px;
            }
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(25, 15, 25, 15)
        header_layout.setSpacing(5)
        
        title = QLabel("AI SMART TECH")
        title.setStyleSheet("""
            color: white;
            font-size: 32px;
            font-weight: bold;
            letter-spacing: 3px;
        """)
        
        subtitle = QLabel("Vision Detection System")
        subtitle.setStyleSheet("""
            color: rgba(255,255,255,0.9);
            font-size: 14px;
            letter-spacing: 1px;
        """)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        left_layout.addWidget(header)
        
        # Mode Toggle
        mode_frame = QFrame()
        mode_frame.setFixedHeight(60)
        mode_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
            }
        """)
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(15, 10, 15, 10)
        mode_layout.setSpacing(10)
        
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("color: #666; font-size: 13px; font-weight: bold;")
        
        self.btn_pills = QPushButton("💊 PILLS")
        self.btn_pills.setFixedSize(120, 40)
        self.btn_pills.setCursor(Qt.PointingHandCursor)
        self.btn_pills.clicked.connect(lambda: self.switch_mode("pills"))
        
        self.btn_boxes = QPushButton("📦 BOXES")
        self.btn_boxes.setFixedSize(120, 40)
        self.btn_boxes.setCursor(Qt.PointingHandCursor)
        self.btn_boxes.clicked.connect(lambda: self.switch_mode("boxes"))
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.btn_pills)
        mode_layout.addWidget(self.btn_boxes)
        mode_layout.addStretch()
        
        left_layout.addWidget(mode_frame)
        
        # Video Display
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background-color: #000;
                border: 3px solid #e0e0e0;
                border-radius: 16px;
            }
        """)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(8, 8, 8, 8)
        
        self.lbl_video = QLabel("Initializing Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(900, 600)
        self.lbl_video.setStyleSheet("""
            background-color: #000;
            color: #666;
            font-size: 16px;
        """)
        video_layout.addWidget(self.lbl_video)
        
        left_layout.addWidget(video_frame, stretch=1)
        
        # Status Bar
        status_frame = QFrame()
        status_frame.setFixedHeight(50)
        status_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
            }
        """)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 10, 20, 10)
        
        self.lbl_status = QLabel("● System Ready")
        self.lbl_status.setStyleSheet("""
            color: #4CAF50;
            font-size: 14px;
            font-weight: bold;
        """)
        
        self.btn_clear = QPushButton("🔄 CLEAR SELECTION")
        self.btn_clear.setFixedSize(150, 30)
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_all_selections)
        self.btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                color: #666;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        status_layout.addWidget(self.lbl_status)
        status_layout.addStretch()
        status_layout.addWidget(self.btn_clear)
        
        left_layout.addWidget(status_frame)
        
        main_layout.addWidget(left_panel, stretch=6)
        
        # === RIGHT: CARDS PANEL ===
        # === RIGHT: CARDS PANEL ===
        right_panel = QFrame()
        right_panel.setFixedWidth(420)
        right_panel.setStyleSheet("""
            QFrame { background-color: white; border-left: 3px solid #e0e0e0; }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(12)
        
        # 1. Detection Header
        detection_header = QFrame()
        detection_header.setFixedHeight(55)
        detection_header.setStyleSheet("background-color: #f5f5f5; border-radius: 10px;")
        detection_header_layout = QHBoxLayout(detection_header)
        detection_header_layout.setContentsMargins(15, 12, 15, 12)
        
        detection_title = QLabel("ยาที่ตรวจพบ")
        detection_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        self.lbl_detection_count = QLabel("0 รายการ")
        self.lbl_detection_count.setStyleSheet("font-size: 14px; font-weight: bold; color: #1976D2;")
        
        detection_header_layout.addWidget(detection_title)
        detection_header_layout.addStretch()
        detection_header_layout.addWidget(self.lbl_detection_count)
        right_layout.addWidget(detection_header)
        
        # 2. Scrollable Cards Area (3-Column Grid)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: white; }
            QScrollBar:vertical { border: none; background: #f5f5f5; width: 6px; border-radius: 3px; }
            QScrollBar::handle:vertical { background: #ccc; border-radius: 3px; }
        """)
        
        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setSpacing(8)
        self.cards_layout.setContentsMargins(3, 3, 3, 3)
        self.cards_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(self.cards_container)
        
        # 3. Summary List Panel
        summary_frame = QFrame()
        summary_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px;")
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(15, 12, 15, 12)
        
        summary_title = QLabel("📋 รายการยาที่ตรวจพบ")
        summary_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #333;")
        summary_layout.addWidget(summary_title)
        
        # สร้าง summary_text ก่อนจะไปตั้งค่า Height
        self.summary_text = QLabel("ยังไม่พบยา")
        self.summary_text.setStyleSheet("""
            background-color: white; color: #333; font-size: 14px;
            padding: 12px; border-radius: 8px; line-height: 1.8;
        """)
        self.summary_text.setWordWrap(True)
        self.summary_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary_text.setMinimumHeight(120)
        self.summary_text.setMaximumHeight(200)
        
        summary_layout.addWidget(self.summary_text)
        
        # 4. Settings Panel (Confidence Slider)
        settings_frame = QFrame()
        settings_frame.setFixedHeight(90)
        settings_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px;")
        settings_layout = QVBoxLayout(settings_frame)
        
        conf_label = QLabel("Confidence Threshold")
        conf_label.setStyleSheet("color: #666; font-size: 11px; font-weight: bold;")
        
        slider_layout = QHBoxLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(30, 95)
        self.slider_conf.setValue(60)
        self.slider_conf.valueChanged.connect(self.update_confidence_label)
        
        self.lbl_conf_value = QLabel("60%")
        self.lbl_conf_value.setFixedWidth(40)
        self.lbl_conf_value.setStyleSheet("font-weight: bold; color: #667eea;")
        
        slider_layout.addWidget(self.slider_conf)
        slider_layout.addWidget(self.lbl_conf_value)
        settings_layout.addWidget(conf_label)
        settings_layout.addLayout(slider_layout)

        # --- จัดลำดับการวางใน Right Panel (Stretch คือหัวใจสำคัญ) ---
        right_layout.addWidget(scroll_area, stretch=4)   # พื้นที่การ์ดใหญ่สุด
        right_layout.addWidget(summary_frame, stretch=2) # พื้นที่สรุปเล็กลงมา
        right_layout.addWidget(settings_frame, stretch=0) # ส่วนตั้งค่าไม่ต้องยืด
        
        main_layout.addWidget(right_panel)
        # Apply global styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        # Update mode buttons
        self.update_mode_buttons()

    def update_mode_buttons(self):
        """Update mode button styles"""
        if self.current_mode == "pills":
            self.btn_pills.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
            self.btn_boxes.setStyleSheet("""
                QPushButton {
                    background-color: #f5f5f5;
                    color: #666;
                    border: 2px solid #e0e0e0;
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)
        else:
            self.btn_boxes.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #f093fb, stop:1 #f5576c);
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
            self.btn_pills.setStyleSheet("""
                QPushButton {
                    background-color: #f5f5f5;
                    color: #666;
                    border: 2px solid #e0e0e0;
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)

    def update_confidence_label(self):
        """Update confidence label"""
        value = self.slider_conf.value()
        self.lbl_conf_value.setText(f"{value}%")

    def update_frame(self):
        """Main update loop"""
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        current_time = time.time()
        should_detect = (current_time - self.last_detection_time) >= self.detection_interval
        
        if should_detect:
            self.last_detection_time = current_time
            self.process_detection(frame)
        else:
            self.show_video(frame)

    def process_detection(self, frame: np.ndarray):
        """Process frame for detection - ไม่วาดกรอบ แต่แสดงใน cards ฝั่งขวา"""
        display = frame.copy()
        h, w = display.shape[:2]
        conf = self.slider_conf.value() / 100.0
        
        try:
            results = self.detector.model(frame, conf=conf, verbose=False)
            raw_inventory = {}
            detected_images = {}
            detected_bboxes = {}
            
            for r in results:
                if not hasattr(r, 'boxes'):
                    continue
                
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                    
                    # Extract crop
                    pad = 10
                    y1_p = max(0, y1 - pad)
                    x1_p = max(0, x1 - pad)
                    y2_p = min(h, y2 + pad)
                    x2_p = min(w, x2 + pad)
                    crop = frame[y1_p:y2_p, x1_p:x2_p]
                    
                    if crop.size == 0:
                        continue
                    
                    # Recognize
                    name, score = None, 0
                    if self.arcface_model and self.db_manager:
                        emb = self.compute_embedding(crop)
                        if emb is not None:
                            name, score = self.db_manager.search(emb)
                    
                    if name and score > conf:
                        # Update raw inventory
                        if name not in raw_inventory:
                            raw_inventory[name] = 0
                        raw_inventory[name] += 1
                        
                        # Store bbox
                        detected_bboxes[name] = (x1, y1, x2, y2)
                        
                        # Store best image
                        if name not in detected_images:
                            detected_images[name] = crop
            
            # *** ANTI-FLICKER: ใช้ tracker เพื่อ smooth detection ***
            self.detection_tracker.update(raw_inventory, detected_bboxes)
            stable_inventory = self.detection_tracker.get_stable_detections()
            
            # Update cards ด้วย stable inventory
            self.update_cards(stable_inventory, detected_images)
            
        except Exception as e:
            pass
        
        self.show_video(display)

    def update_cards(self, inventory: Dict[str, int], images: Dict[str, np.ndarray]):
        """Update detection cards และ summary list"""
        # Remove old cards
        for name in list(self.detection_cards.keys()):
            if name not in inventory:
                card = self.detection_cards[name]
                self.cards_layout.removeWidget(card)
                card.deleteLater()
                del self.detection_cards[name]
                if name in self.detection_images:
                    del self.detection_images[name]
        
        # Update/Add cards
        row, col = 0, 0
        for idx, (name, count) in enumerate(sorted(inventory.items(), key=lambda x: x[1], reverse=True)):
            if name in self.detection_cards:
                # Update existing card
                card = self.detection_cards[name]
                card.item_count = count
                card.lbl_count.setText(f"×{count}")
                
                # Update image if new one available
                if name in images:
                    rgb = cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB)
                    h, w, c = rgb.shape
                    qi = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qi)
                    card.set_image(pixmap)
                    self.detection_images[name] = images[name]
            else:
                # Create new card
                pixmap = None
                if name in images:
                    rgb = cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB)
                    h, w, c = rgb.shape
                    qi = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qi)
                    self.detection_images[name] = images[name]
                
                card = DetectionCard(name, count, pixmap)
                card.clicked = lambda n=name: self.on_card_clicked(n)
                self.detection_cards[name] = card
            
            # Update selection state
            card.set_selected(self.selected_item_name == name.lower())
            
            # Position in grid
            row = idx // 3
            col = idx % 3
            self.cards_layout.addWidget(card, row, col)
        
        # Update count label
        total = len(inventory)
        self.lbl_detection_count.setText(f"{total} รายการ")
        
        # === UPDATE SUMMARY LIST ===
        if inventory:
            summary_lines = []
            total_items = sum(inventory.values())
            
            # เรียงตามจำนวนมากไปน้อย
            sorted_items = sorted(inventory.items(), key=lambda x: x[1], reverse=True)
            
            for name, count in sorted_items:
                # ใช้ emoji indicator
                if self.selected_item_name == name.lower():
                    indicator = "✅"
                    style = "font-weight: bold; color: #1976D2;"
                else:
                    indicator = "•"
                    style = "font-weight: normal; color: #333;"
                
                # แสดงแบบตัวหนา อ่านง่าย
                summary_lines.append(f'<div style="{style}">{indicator} <b>{name.upper()}</b>: {count} รายการ</div>')
            
            # แสดงยอดรวม
            summary_lines.append('<div style="margin-top: 10px; border-top: 2px solid #ddd; padding-top: 10px;"></div>')
            summary_lines.append(f'<div style="font-size: 16px; font-weight: bold; color: #1976D2;">รวมทั้งหมด: {total_items} รายการ</div>')
            
            summary_text = "".join(summary_lines)
            self.summary_text.setText(summary_text)
            self.summary_text.setStyleSheet("""
                background-color: white;
                color: #333;
                font-size: 15px;
                padding: 15px;
                border-radius: 8px;
                line-height: 2.0;
            """)
        else:
            self.summary_text.setText('<div style="color: #999; font-style: italic; text-align: center;">ยังไม่พบยา</div>')
            self.summary_text.setStyleSheet("""
                background-color: white;
                color: #999;
                font-size: 15px;
                padding: 15px;
                border-radius: 8px;
            """)


    def on_card_clicked(self, name: str):
        """Handle card click - TOGGLE selection"""
        name_lower = name.lower()
        
        # Toggle: if already selected, deselect; otherwise select
        if self.selected_item_name == name_lower:
            # Deselect
            self.selected_item_name = None
            self.lbl_status.setText("● System Ready")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        else:
            # Select
            self.selected_item_name = name_lower
            self.lbl_status.setText(f"● Tracking: {name.upper()}")
            self.lbl_status.setStyleSheet("color: #2196F3; font-size: 14px; font-weight: bold;")
        
        # Update all cards
        for card_name, card in self.detection_cards.items():
            card.set_selected(self.selected_item_name == card_name.lower())

    def clear_all_selections(self):
        """Clear all selections"""
        self.selected_item_name = None
        self.lbl_status.setText("● System Ready")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        
        for card in self.detection_cards.values():
            card.set_selected(False)

    def switch_mode(self, mode: str):
        """Switch detection mode"""
        self.current_mode = mode
        self.clear_all_selections()
        self.update_mode_buttons()
        
        # Clear all cards
        for card in self.detection_cards.values():
            self.cards_layout.removeWidget(card)
            card.deleteLater()
        self.detection_cards.clear()
        self.detection_images.clear()
        
        # *** RESET TRACKER ***
        self.detection_tracker = DetectionTracker(memory_frames=5, min_stable_frames=1)
        
        self.load_resources(mode)

    def load_resources(self, mode: str):
        """Load AI resources"""
        self.lbl_status.setText(f"● Loading {mode.upper()}...")
        self.lbl_status.setStyleSheet("color: #FF9800; font-size: 14px; font-weight: bold;")
        
        # Database
        db_path = os.path.join(BASE_DIR, "data", mode)
        os.makedirs(db_path, exist_ok=True)
        db_files = [f for f in os.listdir(db_path) if f.endswith(".db")]
        target_db = os.path.join(db_path, db_files[0]) if db_files else os.path.join(db_path, "default.db")
        
        if self.db_manager:
            try:
                self.db_manager.close()
            except:
                pass
        
        try:
            self.db_manager = DatabaseManager(target_db)
        except:
            self.db_manager = None
        
        # YOLO
        yolo_path = os.path.join(BASE_DIR, "models", "best.onnx")
        if not os.path.exists(yolo_path):
            yolo_path = os.path.join(BASE_DIR, "model_weights", "best.onnx")
        
        if os.path.exists(yolo_path):
            try:
                self.detector.load_model(yolo_path)
            except:
                pass
        
        # ArcFace
        model_name = "best_pill.pth" if mode == "pills" else "best_boxes.pth"
        model_path = os.path.join(BASE_DIR, "models", model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, "model_weights", model_name)
        
        if os.path.exists(model_path):
            try:
                model = PillModel(num_classes=1000, model_name='convnext_small', embed_dim=512)
                ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
                clean_dict = {k: v for k, v in ckpt.items() if not k.startswith('head')}
                model.load_state_dict(clean_dict, strict=False)
                model.to(self.device).eval()
                self.arcface_model = model
                
                self.lbl_status.setText("● System Ready")
                self.lbl_status.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
            except Exception as e:
                self.arcface_model = None
                self.lbl_status.setText("● Model Error")
                self.lbl_status.setStyleSheet("color: #F44336; font-size: 14px; font-weight: bold;")
        else:
            self.arcface_model = None
            self.lbl_status.setText("● Model Missing")
            self.lbl_status.setStyleSheet("color: #F44336; font-size: 14px; font-weight: bold;")

    @torch.no_grad()
    def compute_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Compute embedding"""
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
            embedding = self.arcface_model(tensor).cpu().numpy().flatten()
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except:
            return None

    def show_video(self, frame: np.ndarray):
        """Display video"""
        try:
            h, w, c = frame.shape
            qi = QImage(frame.data, w, h, c * w, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qi)
            scaled = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_video.setPixmap(scaled)
        except:
            pass

    def closeEvent(self, event):
        """Cleanup"""
        try:
            self.camera.stop()
            if self.db_manager:
                self.db_manager.close()
        except:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    
    window = ModernVisionStation()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()