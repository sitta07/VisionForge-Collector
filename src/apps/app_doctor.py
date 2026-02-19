"""
AI SMART TECH — Vision System
Aesthetic Direction: Clinical Noir × Modern Dashboard
[UPDATED]: Pre-pack (Sticker) Mode Ready! 🏷️
"""

import sys
import os
import time
import hashlib
import cv2
import numpy as np
import torch
import subprocess
import sqlite3
from torchvision import transforms
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
import logging
os.environ["FLAGS_enable_pir_api"] = "0"
# ปิด Log จุกจิกของ Paddle
os.environ["GLOG_minloglevel"] = "2"
logging.getLogger('ultralytics').setLevel(logging.ERROR)
os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"
os.environ['YOLO_VERBOSE'] = 'False'
from paddleocr import PaddleOCR

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGridLayout, QSizePolicy,
    QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QBrush
)

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


# ══════════════════════════════════════════════════════════════
#   PALETTE  ·  Clinical Noir
# ══════════════════════════════════════════════════════════════
P = {
    "page":          "#F5F5F2",
    "surface":       "#FFFFFF",
    "surface_raised":"#FAFAF8",
    "surface_dim":   "#EFEFEC",
    "border":        "#E2E2DC",
    "border_med":    "#C8C8C0",
    "ink":           "#141412",
    "ink_2":         "#3A3A36",
    "ink_3":         "#7A7A72",
    "ink_4":         "#AEAEA6",
    "g":             "#0F7B3C",
    "g_bright":      "#12A050",
    "g_glow":        "#1ADF6A",
    "g_light":       "#E6F5EC",
    "g_xlight":      "#F2FAF5",
    "o":             "#C24B00",
    "o_bright":      "#E05800",
    "o_light":       "#FDEEE4",
    "o_xlight":      "#FFF6F0",
    "top_bg":        "#111110",
    "top_border":    "#242422",
    "shadow":        "rgba(0,0,0,0.08)",
}

FONT  = "Plus Jakarta Sans"
MONO  = "IBM Plex Mono"
R     = "12px"
R_LG  = "16px"

_CARD_NAME_NORMAL = f"color:{P['ink_2']};font-family:'{FONT}';font-size:12px;font-weight:700;letter-spacing:0.5px;"
_CARD_NAME_SEL    = f"color:{P['g']};font-family:'{FONT}';font-size:12px;font-weight:700;letter-spacing:0.5px;"
_CARD_CNT_NORMAL  = f"background:{P['o_light']};color:{P['o']};font-family:'{MONO}';font-size:12px;font-weight:700;border-radius:6px;"
_CARD_CNT_SEL     = f"background:{P['o']};color:white;font-family:'{MONO}';font-size:12px;font-weight:700;border-radius:6px;"

# ══════════════════════════════════════════════════════════════
#   DETECTION TRACKER
# ══════════════════════════════════════════════════════════════
class DetectionTracker:
    def __init__(self, memory_frames=5, min_stable_frames=1):
        self.memory_frames     = memory_frames
        self.min_stable_frames = min_stable_frames
        self.detections: Dict  = {}

    def update(self, items: Dict[str, int], bboxes: Dict[str, tuple]):
        now = time.time()
        for name, count in items.items():
            if name not in self.detections:
                self.detections[name] = {
                    'count': count, 'frames_seen': 1,
                    'last_seen_time': now, 'last_bbox': bboxes.get(name)
                }
            else:
                d = self.detections[name]
                d['count']          = count
                d['frames_seen']    = min(d['frames_seen'] + 1, 10)
                d['last_seen_time'] = now
                if name in bboxes:
                    d['last_bbox'] = bboxes[name]

        for name in list(self.detections):
            if name not in items:
                self.detections[name]['frames_seen'] -= 1
                if self.detections[name]['frames_seen'] <= -self.memory_frames:
                    del self.detections[name]

    def get_stable(self) -> Dict[str, int]:
        return {n: max(0, d['count']) for n, d in self.detections.items()
                if d['frames_seen'] >= self.min_stable_frames}

# ══════════════════════════════════════════════════════════════
#   UI COMPONENTS (Dot, Card, Button)
# ══════════════════════════════════════════════════════════════
class LiveDot(QWidget):
    def __init__(self, color: str = P['g_glow']):
        super().__init__()
        self.setFixedSize(10, 10)
        self._color  = QColor(color)
        self._alpha  = 1.0
        self._going  = True
        t = QTimer(self)
        t.timeout.connect(self._pulse)
        t.start(40)

    def set_color(self, c: str):
        self._color = QColor(c)
        self.update()

    def _pulse(self):
        step = 0.04
        if self._going:
            self._alpha = min(1.0, self._alpha + step)
            if self._alpha >= 1.0: self._going = False
        else:
            self._alpha = max(0.2, self._alpha - step)
            if self._alpha <= 0.2: self._going = True
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(self._color)
        c.setAlphaF(self._alpha)
        p.setBrush(QBrush(c))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, 8, 8)

class ItemCard(QFrame):
    def __init__(self, name: str, count: int, image: Optional[QPixmap] = None):
        super().__init__()
        self.item_name   = name
        self.item_count  = count
        self.is_selected = False
        self._click_fn   = None

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(158)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 18))
        self.setGraphicsEffect(shadow)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setFixedHeight(86)
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lbl_image.setStyleSheet(f"""
            background: {P['surface_dim']};
            border-radius: 8px;
            color: {P['ink_4']};
            font-size: 26px;
        """)
        if image:
            self.lbl_image.setPixmap(
                image.scaled(180, 84, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.lbl_image.setText("◈")
        lay.addWidget(self.lbl_image)

        bot = QHBoxLayout()
        bot.setSpacing(6)

        self.lbl_name = QLabel(name.upper())
        self.lbl_name.setWordWrap(False)
        self.lbl_name.setStyleSheet(_CARD_NAME_NORMAL)

        self.lbl_count = QLabel(str(count))
        self.lbl_count.setFixedSize(28, 24)
        self.lbl_count.setAlignment(Qt.AlignCenter)
        self.lbl_count.setStyleSheet(_CARD_CNT_NORMAL)

        bot.addWidget(self.lbl_name, stretch=1)
        bot.addWidget(self.lbl_count)
        lay.addLayout(bot)
        self._apply_style()

    def set_image(self, px: QPixmap):
        self.lbl_image.setPixmap(
            px.scaled(self.lbl_image.width() or 180, 84,
                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def set_selected(self, sel: bool):
        if self.is_selected == sel: return
        self.is_selected = sel
        self._apply_style()

    def _apply_style(self):
        if self.is_selected:
            self.setStyleSheet(f"ItemCard {{ background: {P['g_light']}; border: 2px solid {P['g']}; border-radius: {R_LG}; }}")
            self.lbl_name.setStyleSheet(_CARD_NAME_SEL)
            self.lbl_count.setStyleSheet(_CARD_CNT_SEL)
        else:
            self.setStyleSheet(f"ItemCard {{ background: {P['surface']}; border: 1px solid {P['border']}; border-radius: {R_LG}; }} ItemCard:hover {{ background: {P['g_xlight']}; border-color: {P['border_med']}; }}")
            self.lbl_name.setStyleSheet(_CARD_NAME_NORMAL)
            self.lbl_count.setStyleSheet(_CARD_CNT_NORMAL)

    def mousePressEvent(self, _):
        if self._click_fn: self._click_fn()

class ModeButton(QPushButton):
    def __init__(self, label: str, active: bool = False):
        super().__init__(label)
        self.setFixedSize(128, 38)
        self.setCursor(Qt.PointingHandCursor)
        self.set_active(active)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self.setStyleSheet(f"QPushButton {{ background: {P['g']}; color: white; border: none; border-radius: 10px; font-family: '{FONT}'; font-size: 13px; font-weight: 700; letter-spacing: 0.2px; }}")
        else:
            self.setStyleSheet(f"QPushButton {{ background: transparent; color: {P['ink_3']}; border: 1px solid {P['border_med']}; border-radius: 10px; font-family: '{FONT}'; font-size: 13px; font-weight: 600; }} QPushButton:hover {{ background: {P['g_xlight']}; color: {P['g']}; border-color: {P['g']}; }}")

# ══════════════════════════════════════════════════════════════
#   MAIN WINDOW
# ══════════════════════════════════════════════════════════════
class VisionStation(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI SMART TECH  ·  Vision System")
        self.showMaximized()
        self.setMinimumSize(1100, 720)

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera    = CameraManager()
        self.detector  = ObjectDetector()
        self.processor = ImageProcessor()

        self.arcface_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.arcface_model   = None
        self.db_manager      = None
        
        # 🔥 [EDIT] เริ่มต้นด้วยโหมด prepack
        self.current_mode    = "prepack" 
        self.selected_name   = None
        self.detection_cards:  Dict[str, ItemCard]   = {}
        self.detection_images: Dict[str, np.ndarray] = {}
        self.tracker = DetectionTracker()

        self._executor        = ThreadPoolExecutor(max_workers=1)
        self._detect_future:  Optional[Future] = None
        self._frame_queue:    deque = deque(maxlen=2)
        self._last_inventory: Dict[str, int] = {}
        self._embed_cache:    Dict[str, np.ndarray] = {}
        self._embed_cache_max = 256
        self.latest_boxes: list = []

        # 🔥 [NEW] โหลดรายชื่อยา Pre-pack จาก Database เข้า RAM
        self.known_prepack_drugs = []
        self._load_prepack_db() # เรียกฟังก์ชันที่เราจะสร้างด้านล่าง

        print("📚 Loading PaddleOCR (Stable 2.8.1)...")
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR) 
        
        # 🔥 เพิ่ม use_gpu=False เข้าไป ให้มันรันบน CPU สบายๆ
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)
        self.active_tracks = {}
        self.track_id_counter = 0
        print("✅ OCR Ready!")
        self._build_ui()
        
        self._build_ui()
        self.camera.start(0)
        
        # 🔥 [EDIT] รัน init เริ่มแรกด้วยโหมด prepack
        self.switch_mode("prepack")

        self._vtimer = QTimer()
        self._vtimer.timeout.connect(self._tick_video)
        self._vtimer.start(33)

        self._dtimer = QTimer()
        self._dtimer.timeout.connect(self._tick_detect)
        self._dtimer.start(150)

        self._utimer = QTimer()
        self._utimer.timeout.connect(self._tick_ui)
        self._utimer.start(50)

    # ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet(f"background: {P['page']};")
        self.setCentralWidget(root)

        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)
        vlay.addWidget(self._topbar())

        body = QWidget()
        blay = QHBoxLayout(body)
        blay.setContentsMargins(0, 0, 0, 0)
        blay.setSpacing(0)
        blay.addWidget(self._video_panel(), stretch=1)
        self._sb = self._sidebar()
        blay.addWidget(self._sb, stretch=0)
        vlay.addWidget(body, stretch=1)

    def _topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(58)
        bar.setStyleSheet(f"background: {P['top_bg']}; border-bottom: 1px solid {P['top_border']};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(28, 0, 28, 0)
        lay.setSpacing(0)

        accent = QWidget()
        accent.setFixedSize(3, 28)
        accent.setStyleSheet(f"background: {P['g_glow']}; border-radius: 2px;")
        lay.addWidget(accent)
        lay.addSpacing(14)

        brand = QLabel("AI SMART TECH")
        brand.setStyleSheet(f"color: #FFFFFF; font-family: '{FONT}'; font-size: 16px; font-weight: 800; letter-spacing: 3px;")
        sep = QLabel("  /  ")
        sep.setStyleSheet(f"color: #333330; font-size:16px; font-family:'{MONO}';")
        sub = QLabel("ระบบตรวจนับและจดจำยา")
        sub.setStyleSheet(f"color: #555550; font-family: '{FONT}'; font-size: 14px; font-weight: 500;")

        lay.addWidget(brand)
        lay.addWidget(sep)
        lay.addWidget(sub)
        lay.addStretch()

        status_pill = QWidget()
        status_pill.setFixedHeight(30)
        status_pill.setStyleSheet(f"background: #1E1E1C; border: 1px solid #2A2A28; border-radius: 15px;")
        spill = QHBoxLayout(status_pill)
        spill.setContentsMargins(12, 0, 16, 0)
        spill.setSpacing(8)

        self.status_dot = LiveDot(P['g_glow'])
        self.lbl_status = QLabel("พร้อมใช้งาน")
        self.lbl_status.setStyleSheet(f"color: #888882; font-family: '{FONT}'; font-size: 13px;")
        spill.addWidget(self.status_dot)
        spill.addWidget(self.lbl_status)

        lay.addWidget(status_pill)
        return bar

    def _video_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {P['page']};")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(24, 20, 16, 20)
        lay.setSpacing(14)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        mode_lbl = QLabel("Detection Mode")
        mode_lbl.setStyleSheet(f"color: {P['ink_4']}; font-family: '{MONO}'; font-size: 11px; font-weight: 600; letter-spacing: 1px;")

        # 🔥 [EDIT] ปรับ UI ปุมเป็นโหมดซองยา
        self.btn_prepack = ModeButton("🏷️  ซองจัดยา", True)
        self.btn_boxes = ModeButton("📦  กล่องยา", False)
        self.btn_prepack.clicked.connect(lambda: self.switch_mode("prepack"))
        self.btn_boxes.clicked.connect(lambda: self.switch_mode("boxes"))

        ctrl.addWidget(mode_lbl)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.btn_prepack)
        ctrl.addWidget(self.btn_boxes)
        ctrl.addStretch()

        self.mode_tag = QLabel("MODE: PRE-PACK")
        self.mode_tag.setStyleSheet(f"color: {P['g']}; font-family: '{MONO}'; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; background: {P['g_light']}; padding: 4px 12px; border-radius: 6px;")
        ctrl.addWidget(self.mode_tag)

        lay.addLayout(ctrl)

        self.lbl_video = QLabel("กำลังเชื่อมต่อกล้อง...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet(f"background: #0A0B0A; border-radius: {R_LG}; color: {P['ink_4']}; font-family: '{FONT}'; font-size: 15px;")
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        vid_shadow = QGraphicsDropShadowEffect(self.lbl_video)
        vid_shadow.setBlurRadius(24)
        vid_shadow.setOffset(0, 4)
        vid_shadow.setColor(QColor(0, 0, 0, 40))
        self.lbl_video.setGraphicsEffect(vid_shadow)

        lay.addWidget(self.lbl_video, stretch=1)
        lay.addWidget(self._action_bar())
        return panel

    def _action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setStyleSheet(f"background: {P['surface']}; border-radius: {R}; border: none;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(0)

        self.lbl_action_mode = QLabel("ซองจัดยา · ระบบพร้อม")
        self.lbl_action_mode.setStyleSheet(f"color: {P['ink_3']}; font-family: '{FONT}'; font-size: 14px;")
        lay.addWidget(self.lbl_action_mode)
        lay.addStretch()

        self.btn_clear = QPushButton("ล้างการเลือก")
        self.btn_clear.setFixedSize(120, 32)
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_all_selections)
        self.btn_clear.setStyleSheet(f"QPushButton {{ background: transparent; border: 1px solid {P['border_med']}; border-radius: 8px; color: {P['ink_3']}; font-family: '{FONT}'; font-size: 13px; font-weight: 600; }} QPushButton:hover {{ background: {P['o_xlight']}; border-color: {P['o']}; color: {P['o']}; }}")
        lay.addWidget(self.btn_clear)
        return bar

    # ───────────────────────────────────────────────────────
    def _sidebar(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {P['surface']}; border-left: 1px solid {P['border']};")
        sw = QApplication.primaryScreen().availableGeometry().width()
        w.setFixedWidth(max(340, min(460, int(sw * 0.26))))

        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(self._sb_header())
        lay.addWidget(self._cards_scroll(), stretch=2)
        lay.addWidget(self._summary_panel(), stretch=3)
        return w

    def _sb_header(self) -> QWidget:
        h = QWidget()
        h.setFixedHeight(58)
        h.setStyleSheet(f"background: {P['surface']};")
        lay = QHBoxLayout(h)
        lay.setContentsMargins(22, 0, 22, 0)

        label = QLabel("รายการที่พบ")
        label.setStyleSheet(f"color: {P['ink']}; font-family: '{FONT}'; font-size: 17px; font-weight: 700;")

        self.lbl_count = QLabel("0")
        self.lbl_count.setFixedHeight(28)
        self.lbl_count.setStyleSheet(f"background: {P['o_light']}; color: {P['o']}; font-family: '{MONO}'; font-size: 15px; font-weight: 800; padding: 0px 14px; border-radius: 14px;")

        lay.addWidget(label)
        lay.addStretch()
        lay.addWidget(self.lbl_count)
        return h

    def _cards_scroll(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {P['surface_dim']}; border-top: 1px solid {P['border']}; border-bottom: 1px solid {P['border']}; }} QScrollBar:vertical {{ border: none; background: transparent; width: 4px; }} QScrollBar::handle:vertical {{ background: {P['border_med']}; border-radius: 2px; min-height: 20px; }} QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}")
        self.cards_container = QWidget()
        self.cards_container.setStyleSheet(f"background: {P['surface_dim']};")
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setSpacing(10)
        self.cards_layout.setContentsMargins(14, 14, 14, 14)
        self.cards_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self.cards_container)
        return scroll

    def _summary_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {P['surface']};")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(22, 22, 22, 26)
        lay.setSpacing(0)

        hdr = QHBoxLayout()
        hdr.setSpacing(12)

        accent = QWidget()
        accent.setFixedSize(4, 22)
        accent.setStyleSheet(f"background: {P['o']}; border-radius: 2px;")

        title = QLabel("สรุปรายการ")
        title.setStyleSheet(f"color: {P['ink']}; font-family: '{FONT}'; font-size: 18px; font-weight: 700;")
        hdr.addWidget(accent, alignment=Qt.AlignVCenter)
        hdr.addWidget(title)
        hdr.addStretch()
        lay.addLayout(hdr)
        lay.addSpacing(16)

        self.summary_text = QLabel("ยังไม่พบรายการ")
        self.summary_text.setWordWrap(True)
        self.summary_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary_text.setStyleSheet(f"color: {P['ink_4']}; font-family: '{FONT}'; font-size: 18px; font-style: italic;")
        self.summary_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.summary_text, stretch=1)

        lay.addSpacing(20)

        total_bg = QWidget()
        total_bg.setStyleSheet(f"background: {P['o_xlight']}; border-radius: {R}; border: 1px solid {P['o_light']};")
        tblay = QHBoxLayout(total_bg)
        tblay.setContentsMargins(18, 14, 18, 14)

        lbl_t = QLabel("รวมทั้งหมด")
        lbl_t.setStyleSheet(f"color: {P['ink_3']}; font-family: '{FONT}'; font-size: 15px; font-weight: 500;")

        num_row = QHBoxLayout()
        num_row.setSpacing(6)
        num_row.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.lbl_total = QLabel("—")
        self.lbl_total.setStyleSheet(f"color: {P['o']}; font-family: '{MONO}'; font-size: 40px; font-weight: 700; line-height: 1;")
        lbl_unit = QLabel("รายการ")
        lbl_unit.setStyleSheet(f"color: {P['ink_3']}; font-family: '{FONT}'; font-size: 15px; font-weight: 500;")
        lbl_unit.setAlignment(Qt.AlignBottom)
        num_row.addWidget(self.lbl_total)
        num_row.addWidget(lbl_unit, alignment=Qt.AlignBottom)

        tblay.addWidget(lbl_t, alignment=Qt.AlignVCenter)
        tblay.addLayout(num_row)
        lay.addWidget(total_bg)

        return panel
    def _load_prepack_db(self):
        """ดึงชื่อยาจาก SQLite มาเก็บไว้ใน RAM (self.known_prepack_drugs)"""
        db_path = os.path.join(BASE_DIR, "data", "prepack", "prepack_drugs.db")
        if not os.path.exists(db_path):
            self.known_prepack_drugs = []
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT name FROM drugs")
            self.known_prepack_drugs = [row[0] for row in cursor.fetchall()]
            conn.close()
            print(f"📦 Loaded {len(self.known_prepack_drugs)} drugs to RAM.")
        except Exception as e:
            print(f"⚠️ Error loading prepack DB: {e}")
            self.known_prepack_drugs = []

    def _set_status(self, text: str, color: str = None):
        self.lbl_status.setText(text)
        self.status_dot.set_color(color or P['g_glow'])

    def _update_mode_btns(self):
        # 🔥 [EDIT] 
        self.btn_prepack.set_active(self.current_mode == "prepack")
        self.btn_boxes.set_active(self.current_mode == "boxes")

    def _tick_video(self):
        frame = self.camera.get_frame()
        if frame is None: return
        self._frame_queue.append(frame)
        self._show_video(frame)

    def _tick_detect(self):
        if self._detect_future is not None and not self._detect_future.done(): return
        if not self._frame_queue: return
        frame = self._frame_queue[-1]
        self._detect_future = self._executor.submit(self._detect_worker, frame.copy())

    def _tick_ui(self):
        if self._detect_future is None or not self._detect_future.done(): return
        try:
            inventory, images, boxes = self._detect_future.result()
            self.latest_boxes = boxes
        except Exception:
            return
        finally:
            self._detect_future = None

        # 🔥 [SENIOR FIX]: ให้อัปเดต Tracker เสมอ! ห้าม return หนี
        # เพื่อให้ระบบนับถอยหลังและ "ลืม" ของที่หายไปจากหน้าจอได้
        self.tracker.update(inventory, {})
        stable = self.tracker.get_stable()

        # 🔥 เทียบความเปลี่ยนแปลงจากค่า stable แทนที่จะเป็นค่าดิบ 
        # (ถ้าค่าบนจอเหมือนเดิมเป๊ะๆ ถึงจะค่อยข้ามการวาด UI)
        if hasattr(self, '_last_stable') and stable == self._last_stable and not images:
            return
            
        self._last_stable = dict(stable)
        self._update_cards(stable, images)
    def _calc_iou(self, boxA, boxB):
        """คำนวณพื้นที่ทับซ้อน (Intersection over Union) เพื่อดูว่าเป็นของชิ้นเดิมไหม"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

    def _detect_worker(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        raw, imgs = {}, {}
        boxes = []
        
        CONF_YOLO = 0.50 
        CONF_RECOG = 0.75 
        IGNORED_CLASSES = ["_BG", "_HAND", "_JUNK", "background"]

        try:
            results = self.detector.model(frame, conf=CONF_YOLO, iou=0.45, verbose=False)
            current_yolo_boxes = []
            
            # 1. รวบรวมกล่องที่ YOLO เจอ
            for r in results:
                if not hasattr(r, 'boxes'): continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    if (x2 - x1) < 20 or (y2 - y1) < 20: continue
                    p = 10
                    crop = frame[max(0, y1-p):min(h, y2+p), max(0, x1-p):min(w, x2+p)]
                    if crop.size > 0:
                        current_yolo_boxes.append({'coords': (x1, y1, x2, y2), 'crop': crop})

            new_tracks = {}
            drugs_list = getattr(self, 'known_prepack_drugs', [])
            used_tracks = set()

            # 2. ประมวลผลแต่ละกล่อง
            for item in current_yolo_boxes:
                coords = item['coords']
                crop = item['crop']
                
                name = "Unknown"
                score = 0.0
                needs_ocr = True 
                track_id = None
                
                if self.current_mode == "prepack":
                    best_iou = 0
                    best_id = None
                    for tid, track in self.active_tracks.items():
                        if tid in used_tracks: continue
                        iou = self._calc_iou(coords, track['coords'])
                        if iou > best_iou:
                            best_iou = iou
                            best_id = tid
                            
                    if best_iou > 0.4 and best_id is not None:
                        cached_name = self.active_tracks[best_id]['name']
                        cached_score = self.active_tracks[best_id]['score']
                        track_id = best_id
                        used_tracks.add(best_id)
                        
                        if cached_name != "Unknown" and not str(cached_name).startswith("?"):
                            name = cached_name
                            score = cached_score
                            needs_ocr = False 
                    else:
                        track_id = self.track_id_counter
                        self.track_id_counter += 1

                    if needs_ocr and hasattr(self, 'ocr'):
                        # ฟังก์ชันย่อยสำหรับรัน OCR
                        def run_ocr(img_crop):
                            res = self.ocr.ocr(img_crop, cls=True)
                            txt = ""
                            if res and res[0]:
                                for line in res[0]:
                                    txt += line[1][0].upper() + " "
                            return txt.strip()
                        
                        # ลองอ่านแบบปกติก่อน
                        extracted_text = run_ocr(crop)
                        best_match = "Unknown"
                        best_score = 0
                        
                        from rapidfuzz import process, fuzz
                        
                        if extracted_text and drugs_list:
                            res_normal = process.extractOne(extracted_text, drugs_list, scorer=fuzz.partial_ratio)
                            if res_normal:
                                best_match, best_score, _ = res_normal

                        # ถ้าอ่านไม่ค่อยออก หรือคะแนนต่ำ ให้หมุนภาพ 180 องศาแล้วอ่านใหม่
                        if not extracted_text or best_score < 70:
                            crop_180 = cv2.rotate(crop, cv2.ROTATE_180) 
                            ext_180 = run_ocr(crop_180)
                            
                            if ext_180 and drugs_list:
                                res_180 = process.extractOne(ext_180, drugs_list, scorer=fuzz.partial_ratio)
                                if res_180 and res_180[1] > best_score:
                                    extracted_text = ext_180
                                    best_match, best_score, _ = res_180

                        # สรุปผล
                        if extracted_text:
                            if best_score >= 70:
                                name = best_match
                                score = best_score / 100.0 
                            else:
                                name = f"? {extracted_text[:12]}" 
                                score = best_score / 100.0
                        else:
                            name = "? (Unreadable)"
                            score = 0.5
                                
                    new_tracks[track_id] = {'coords': coords, 'name': name, 'score': score}
                    
                else:
                    # โหมดกล่องยา
                    if self.arcface_model and self.db_manager:
                        emb = self._embed_cached(crop) 
                        if emb is not None:
                            name, score = self.db_manager.search(emb)
                            
                # --- จัดการ UI ---
                if name in IGNORED_CLASSES: continue
                if self.current_mode != "prepack" and score < CONF_RECOG:
                    name = "Unknown"
                    score = 0.0 
                
                # 🔥 [แก้ปัญหาการ์ดค้าง 100%]: 
                # ให้เพิ่มข้อมูลลง `raw` เฉพาะของที่มองเห็นอยู่ในเฟรมนี้เท่านั้น!
                if name != "Unknown" and not str(name).startswith("?"):
                    raw[name] = raw.get(name, 0) + 1
                    if name not in imgs:
                        imgs[name] = crop
                
                boxes.append({"coords": coords, "name": name, "score": score})

            # 🔥 [แก้ปัญหาการ์ดค้าง 100%]: 
            # ถ้าโหมดซองยา ให้เซฟเฉพาะรายการใหม่ที่เห็นในเฟรมนี้ 
            # ของที่หายไปแล้วจะไม่ถูกจำไว้ใน active_tracks อีก
            if self.current_mode == "prepack":
                self.active_tracks = new_tracks
                
        except Exception as e:
            print(f"Worker Error: {e}")
            import traceback
            traceback.print_exc()
        
        return raw, imgs, boxes
    def _embed_cached(self, img: np.ndarray) -> Optional[np.ndarray]:
        small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_NEAREST)
        key   = hashlib.md5(small.tobytes()).hexdigest()
        if key in self._embed_cache: return self._embed_cache[key]
        emb = self._embed(img)
        if emb is not None:
            if len(self._embed_cache) >= self._embed_cache_max:
                self._embed_cache.pop(next(iter(self._embed_cache)))
            self._embed_cache[key] = emb
        return emb

    def _update_cards(self, inventory: Dict[str, int], images: Dict[str, np.ndarray]):
        for name in list(self.detection_cards):
            if name not in inventory:
                card = self.detection_cards.pop(name)
                self.cards_layout.removeWidget(card)
                card.deleteLater()
                self.detection_images.pop(name, None)

        sorted_items = sorted(inventory.items(), key=lambda x: x[1], reverse=True)
        cols = 2

        for idx, (name, count) in enumerate(sorted_items):
            if name in self.detection_cards:
                card = self.detection_cards[name]
                if card.item_count != count:
                    card.item_count = count
                    card.lbl_count.setText(str(count))
                if name in images:
                    rgb = cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB)
                    hh, ww, cc = rgb.shape
                    qi = QImage(rgb.data, ww, hh, cc * ww, QImage.Format_RGB888)
                    card.set_image(QPixmap.fromImage(qi))
                    self.detection_images[name] = images[name]
            else:
                px = None
                if name in images:
                    rgb = cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB)
                    hh, ww, cc = rgb.shape
                    qi = QImage(rgb.data, ww, hh, cc * ww, QImage.Format_RGB888)
                    px = QPixmap.fromImage(qi)
                    self.detection_images[name] = images[name]
                card = ItemCard(name, count, px)
                card._click_fn = lambda n=name: self._on_card(n)
                self.detection_cards[name] = card

            card.set_selected(self.selected_name == name.lower())
            self.cards_layout.addWidget(card, idx // cols, idx % cols)

        self.lbl_count.setText(str(len(inventory)) if inventory else "0")

        if inventory:
            lines = []
            for name, count in sorted_items:
                sel = self.selected_name == name.lower()
                if sel:
                    row = f'<span style="color:{P["o"]};font-size:9px;">▶</span>&nbsp;<span style="color:{P["g"]};font-family:{FONT};font-size:20px;font-weight:700;">{name.upper()}</span>&nbsp;&nbsp;<span style="color:{P["o"]};font-family:{MONO};font-size:20px;font-weight:700;">{count}</span><span style="color:{P["ink_3"]};font-family:{FONT};font-size:16px;"> รายการ</span>'
                else:
                    row = f'<span style="color:{P["ink_4"]};font-size:9px;">▸</span>&nbsp;<span style="color:{P["ink_2"]};font-family:{FONT};font-size:20px;font-weight:600;">{name.upper()}</span>&nbsp;&nbsp;<span style="color:{P["o_bright"]};font-family:{MONO};font-size:20px;font-weight:700;">{count}</span><span style="color:{P["ink_4"]};font-family:{FONT};font-size:16px;"> รายการ</span>'
                lines.append(row)

            self.summary_text.setText("<br>".join(lines))
            self.summary_text.setStyleSheet(f"color: {P['ink_2']}; font-family: '{FONT}'; font-size: 20px; font-style: normal; line-height: 2;")
            self.lbl_total.setText(str(sum(inventory.values())))
        else:
            self.summary_text.setText("ยังไม่พบรายการ")
            self.summary_text.setStyleSheet(f"color: {P['ink_4']}; font-family: '{FONT}'; font-size: 18px; font-style: italic;")
            self.lbl_total.setText("—")

    def _on_card(self, name: str):
        nl = name.lower()
        mode_th = "ซองจัดยา" if self.current_mode == "prepack" else "กล่องยา"
        if self.selected_name == nl:
            self.selected_name = None
            self._set_status("พร้อมใช้งาน", P['g_glow'])
            self.lbl_action_mode.setText(f"{mode_th} · ระบบพร้อม")
        else:
            self.selected_name = nl
            self._set_status(f"กำลังติดตาม  {name.upper()}", P['o_bright'])
            self.lbl_action_mode.setText(f"กำลังติดตาม:  {name.upper()}")
            self.lbl_action_mode.setStyleSheet(f"color: {P['o']}; font-family: '{FONT}'; font-size: 14px; font-weight: 600;")
        for cn, card in self.detection_cards.items():
            card.set_selected(self.selected_name == cn.lower())

    def clear_all_selections(self):
        self.selected_name = None
        self._set_status("พร้อมใช้งาน", P['g_glow'])
        mode_th = "ซองจัดยา" if self.current_mode == "prepack" else "กล่องยา"
        self.lbl_action_mode.setText(f"{mode_th} · ระบบพร้อม")
        self.lbl_action_mode.setStyleSheet(f"color: {P['ink_3']}; font-family: '{FONT}'; font-size: 14px;")
        for card in self.detection_cards.values():
            card.set_selected(False)

    def _set_hardware_zoom(self, value: int):
        try:
            cmd = ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl", f"zoom_absolute={value}"]
            subprocess.run(cmd, check=False, capture_output=True)
        except Exception:
            pass 

    def switch_mode(self, mode: str):
        self.current_mode = mode
        self.clear_all_selections()
        self._update_mode_btns()
        
        # 🔥 [EDIT] ปรับ UI Text และ Hardware Zoom ให้เข้ากับซองยา
        mode_th = "ซองจัดยา" if mode == "prepack" else "กล่องยา"
        self.mode_tag.setText(f"MODE: {'PRE-PACK' if mode == 'prepack' else 'BOXES'}")
        self.lbl_action_mode.setText(f"{mode_th} · ระบบพร้อม")
        
        if mode == "prepack":
            self._set_hardware_zoom(40) # ถอยซูมออกหน่อย เพราะสติกเกอร์ซองยาใหญ่กว่าเม็ดยา
        else:
            self._set_hardware_zoom(50)

        for card in self.detection_cards.values():
            self.cards_layout.removeWidget(card)
            card.deleteLater()
        self.detection_cards.clear()
        self.detection_images.clear()
        self._last_inventory.clear()
        self._embed_cache.clear()
        self.tracker = DetectionTracker()
        self._load_resources(mode)

    def _load_resources(self, mode: str):
        mode_th = "ซองจัดยา" if mode == "prepack" else "กล่องยา"
        self._set_status(f"กำลังโหลด {mode_th}...", P['o_bright'])

        db_path = os.path.join(BASE_DIR, "data", mode)
        os.makedirs(db_path, exist_ok=True)
        db_files = [f for f in os.listdir(db_path) if f.endswith(".db")]
        target_db = os.path.join(db_path, db_files[0]) if db_files else \
                    os.path.join(db_path, "default.db")

        if self.db_manager:
            try: self.db_manager.close()
            except: pass
        try: self.db_manager = DatabaseManager(target_db)
        except: self.db_manager = None

        # 🔥 [EDIT] ชี้เป้าไปที่โมเดล prepack.onnx ที่เพิ่งเทรนมาใหม่
        yolo_model_name = "prepack.onnx" if mode == "prepack" else "best_doctor.onnx"
        for base in ["models", "model_weights"]:
            p = os.path.join(BASE_DIR, base, yolo_model_name)
            if os.path.exists(p):
                try: self.detector.load_model(p)
                except: pass
                break

        # ArcFace Load (เผื่อไว้ใช้ดึง Vector รูปสติกเกอร์ไปก่อน)
        mname = "best_pills.pth" if mode == "prepack" else "best_boxes.pth"
        mp = None
        for base in ["models", "model_weights"]:
            p = os.path.join(BASE_DIR, base, mname)
            if os.path.exists(p): mp = p; break

        if mp:
            try:
                model = PillModel(num_classes=1000, model_name='convnext_small', embed_dim=512, use_cbam=False)
                ckpt  = torch.load(mp, map_location=self.device, weights_only=True)
                clean = {k: v for k, v in ckpt.items() if not k.startswith('head') and not k.startswith('module.')}
                model.load_state_dict(clean, strict=False)
                model.to(self.device).eval()
                self.arcface_model = model
                self._set_status("พร้อมใช้งาน", P['g_glow'])
            except:
                self.arcface_model = None
                self._set_status("โหลดโมเดลไม่สำเร็จ", "#E03030")
        else:
            self.arcface_model = None
            self._set_status("ไม่พบไฟล์โมเดล", "#E03030")

    @torch.no_grad()
    def _embed(self, img: np.ndarray) -> Optional[np.ndarray]:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t   = self.arcface_transform(rgb).unsqueeze(0).to(self.device)
            
            features = self.arcface_model(t)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            emb = features.cpu().numpy().flatten()
            return emb
        except Exception as e:
            return None

    def _show_video(self, frame: np.ndarray):
        try:
            if hasattr(self, 'latest_boxes') and self.latest_boxes:
                color_bgr = (106, 223, 26) 
                
                for item in self.latest_boxes:
                    x1, y1, x2, y2 = item['coords']
                    name = item['name']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                    
                    label = name.upper()
                    font_scale = 0.5
                    font_thick = 1
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                    
                    cv2.rectangle(frame, (x1, y1 - 22), (x1 + w_text + 10, y1), color_bgr, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 6), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), font_thick, cv2.LINE_AA)

            h, w, c = frame.shape
            qi = QImage(frame.data, w, h, c * w, QImage.Format_BGR888)
            self.lbl_video.setPixmap(
                QPixmap.fromImage(qi).scaled(
                    self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        except Exception: 
            pass

    def closeEvent(self, event):
        self._vtimer.stop()
        self._dtimer.stop()
        self._utimer.stop()
        self._executor.shutdown(wait=False)
        try:
            self.camera.stop()
            if self.db_manager: self.db_manager.close()
        except: pass
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Plus Jakarta Sans", 11))
    app.setStyleSheet(f"QMainWindow, QWidget {{ font-family: 'Plus Jakarta Sans', 'DM Sans', system-ui; }}")
    w = VisionStation()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()