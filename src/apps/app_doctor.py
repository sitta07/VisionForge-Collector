"""
AI SMART TECH — Vision System
Aesthetic Direction: Clinical Noir × Modern Dashboard
ทันสมัย · อ่านง่าย · ดู Production

OPTIMIZATIONS:
- Background thread for detection (ThreadPoolExecutor)
- Frame queue with drop policy (no stale frame buildup)
- Cached BGR→RGB conversion + QImage reuse
- LRU embedding cache (avoids re-embedding same crop hash)
- Batched layout updates (single pass, no repeated removeWidget)
- Summary HTML rebuilt only when inventory changes
- Pre-computed stable dict diff (skip UI update if unchanged)
- Reduced timer granularity: video@30fps, detect@independent thread
"""

import sys
import os
import time
import hashlib
import cv2
import numpy as np
import torch
import subprocess
from torchvision import transforms
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
from functools import lru_cache
import logging

logging.getLogger('ultralytics').setLevel(logging.ERROR)
os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"
os.environ['YOLO_VERBOSE'] = 'False'

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGridLayout, QSizePolicy,
    QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QBrush,
    QLinearGradient, QPen, QPainterPath
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
CONF  = 0.30

# Pre-build common style strings once (avoid repeated f-string eval)
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
#   LIVE DOT
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


# ══════════════════════════════════════════════════════════════
#   ITEM CARD
# ══════════════════════════════════════════════════════════════
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
        if self.is_selected == sel:   # ← skip redundant style apply
            return
        self.is_selected = sel
        self._apply_style()

    def _apply_style(self):
        if self.is_selected:
            self.setStyleSheet(f"""
                ItemCard {{
                    background: {P['g_light']};
                    border: 2px solid {P['g']};
                    border-radius: {R_LG};
                }}
            """)
            self.lbl_name.setStyleSheet(_CARD_NAME_SEL)
            self.lbl_count.setStyleSheet(_CARD_CNT_SEL)
        else:
            self.setStyleSheet(f"""
                ItemCard {{
                    background: {P['surface']};
                    border: 1px solid {P['border']};
                    border-radius: {R_LG};
                }}
                ItemCard:hover {{
                    background: {P['g_xlight']};
                    border-color: {P['border_med']};
                }}
            """)
            self.lbl_name.setStyleSheet(_CARD_NAME_NORMAL)
            self.lbl_count.setStyleSheet(_CARD_CNT_NORMAL)

    def mousePressEvent(self, _):
        if self._click_fn:
            self._click_fn()


# ══════════════════════════════════════════════════════════════
#   MODE BUTTON
# ══════════════════════════════════════════════════════════════
class ModeButton(QPushButton):
    def __init__(self, label: str, active: bool = False):
        super().__init__(label)
        self.setFixedSize(128, 38)
        self.setCursor(Qt.PointingHandCursor)
        self.set_active(active)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {P['g']};
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-family: '{FONT}';
                    font-size: 13px;
                    font-weight: 700;
                    letter-spacing: 0.2px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {P['ink_3']};
                    border: 1px solid {P['border_med']};
                    border-radius: 10px;
                    font-family: '{FONT}';
                    font-size: 13px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: {P['g_xlight']};
                    color: {P['g']};
                    border-color: {P['g']};
                }}
            """)


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
        self.current_mode    = "pills"
        self.selected_name   = None
        self.detection_cards:  Dict[str, ItemCard]   = {}
        self.detection_images: Dict[str, np.ndarray] = {}
        self.tracker = DetectionTracker()

        # ── OPTIMIZATION: background detection thread ──────────────
        self._executor        = ThreadPoolExecutor(max_workers=1)
        self._detect_future:  Optional[Future] = None
        self._frame_queue:    deque = deque(maxlen=2)   # drop stale frames
        self._last_inventory: Dict[str, int] = {}       # diff guard
        self._embed_cache:    Dict[str, np.ndarray] = {}  # hash→embedding
        self._embed_cache_max = 256
        # ──────────────────────────────────────────────────────────

        self._build_ui()
        self.camera.start(0)
        self.switch_mode("pills")

        # Video display timer: ~30 fps
        self._vtimer = QTimer()
        self._vtimer.timeout.connect(self._tick_video)
        self._vtimer.start(33)

        # Detection submission timer: every 150 ms
        self._dtimer = QTimer()
        self._dtimer.timeout.connect(self._tick_detect)
        self._dtimer.start(150)

        # UI update timer: poll result every 50 ms
        self._utimer = QTimer()
        self._utimer.timeout.connect(self._tick_ui)
        self._utimer.start(50)

        # Pending result from background thread
        self._pending_inventory: Optional[Dict[str, int]] = None
        self._pending_images:    Optional[Dict[str, np.ndarray]] = None

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

    # ───────────────────────────────────────────────────────
    def _topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(58)
        bar.setStyleSheet(f"""
            background: {P['top_bg']};
            border-bottom: 1px solid {P['top_border']};
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(28, 0, 28, 0)
        lay.setSpacing(0)

        accent = QWidget()
        accent.setFixedSize(3, 28)
        accent.setStyleSheet(f"background: {P['g_glow']}; border-radius: 2px;")
        lay.addWidget(accent)
        lay.addSpacing(14)

        brand = QLabel("AI SMART TECH")
        brand.setStyleSheet(f"""
            color: #FFFFFF;
            font-family: '{FONT}';
            font-size: 16px;
            font-weight: 800;
            letter-spacing: 3px;
        """)
        sep = QLabel("  /  ")
        sep.setStyleSheet(f"color: #333330; font-size:16px; font-family:'{MONO}';")
        sub = QLabel("ระบบตรวจนับและจดจำยา")
        sub.setStyleSheet(f"""
            color: #555550;
            font-family: '{FONT}';
            font-size: 14px;
            font-weight: 500;
        """)

        lay.addWidget(brand)
        lay.addWidget(sep)
        lay.addWidget(sub)
        lay.addStretch()

        status_pill = QWidget()
        status_pill.setFixedHeight(30)
        status_pill.setStyleSheet(f"""
            background: #1E1E1C;
            border: 1px solid #2A2A28;
            border-radius: 15px;
        """)
        spill = QHBoxLayout(status_pill)
        spill.setContentsMargins(12, 0, 16, 0)
        spill.setSpacing(8)

        self.status_dot = LiveDot(P['g_glow'])
        self.lbl_status = QLabel("พร้อมใช้งาน")
        self.lbl_status.setStyleSheet(f"""
            color: #888882;
            font-family: '{FONT}';
            font-size: 13px;
        """)
        spill.addWidget(self.status_dot)
        spill.addWidget(self.lbl_status)

        lay.addWidget(status_pill)
        return bar

    # ───────────────────────────────────────────────────────
    def _video_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {P['page']};")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(24, 20, 16, 20)
        lay.setSpacing(14)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        mode_lbl = QLabel("Detection Mode")
        mode_lbl.setStyleSheet(f"""
            color: {P['ink_4']};
            font-family: '{MONO}';
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
        """)

        self.btn_pills = ModeButton("💊  ยาเม็ด", True)
        self.btn_boxes = ModeButton("📦  กล่องยา", False)
        self.btn_pills.clicked.connect(lambda: self.switch_mode("pills"))
        self.btn_boxes.clicked.connect(lambda: self.switch_mode("boxes"))

        ctrl.addWidget(mode_lbl)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.btn_pills)
        ctrl.addWidget(self.btn_boxes)
        ctrl.addStretch()

        self.mode_tag = QLabel("MODE: PILLS")
        self.mode_tag.setStyleSheet(f"""
            color: {P['g']};
            font-family: '{MONO}';
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            background: {P['g_light']};
            padding: 4px 12px;
            border-radius: 6px;
        """)
        ctrl.addWidget(self.mode_tag)

        lay.addLayout(ctrl)

        self.lbl_video = QLabel("กำลังเชื่อมต่อกล้อง...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet(f"""
            background: #0A0B0A;
            border-radius: {R_LG};
            color: {P['ink_4']};
            font-family: '{FONT}';
            font-size: 15px;
        """)
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
        bar.setStyleSheet(f"""
            background: {P['surface']};
            border-radius: {R};
            border: none;
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(0)

        self.lbl_action_mode = QLabel("ยาเม็ด · ระบบพร้อม")
        self.lbl_action_mode.setStyleSheet(f"""
            color: {P['ink_3']};
            font-family: '{FONT}';
            font-size: 14px;
        """)
        lay.addWidget(self.lbl_action_mode)
        lay.addStretch()

        self.btn_clear = QPushButton("ล้างการเลือก")
        self.btn_clear.setFixedSize(120, 32)
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_all_selections)
        self.btn_clear.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {P['border_med']};
                border-radius: 8px;
                color: {P['ink_3']};
                font-family: '{FONT}';
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {P['o_xlight']};
                border-color: {P['o']};
                color: {P['o']};
            }}
        """)
        lay.addWidget(self.btn_clear)
        return bar

    # ───────────────────────────────────────────────────────
    def _sidebar(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"""
            background: {P['surface']};
            border-left: 1px solid {P['border']};
        """)
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
        label.setStyleSheet(f"""
            color: {P['ink']};
            font-family: '{FONT}';
            font-size: 17px;
            font-weight: 700;
        """)

        self.lbl_count = QLabel("0")
        self.lbl_count.setFixedHeight(28)
        self.lbl_count.setStyleSheet(f"""
            background: {P['o_light']};
            color: {P['o']};
            font-family: '{MONO}';
            font-size: 15px;
            font-weight: 800;
            padding: 0px 14px;
            border-radius: 14px;
        """)

        lay.addWidget(label)
        lay.addStretch()
        lay.addWidget(self.lbl_count)
        return h

    def _cards_scroll(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {P['surface_dim']};
                border-top: 1px solid {P['border']};
                border-bottom: 1px solid {P['border']};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {P['border_med']};
                border-radius: 2px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
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
        title.setStyleSheet(f"""
            color: {P['ink']};
            font-family: '{FONT}';
            font-size: 18px;
            font-weight: 700;
        """)
        hdr.addWidget(accent, alignment=Qt.AlignVCenter)
        hdr.addWidget(title)
        hdr.addStretch()
        lay.addLayout(hdr)
        lay.addSpacing(16)

        self.summary_text = QLabel("ยังไม่พบรายการ")
        self.summary_text.setWordWrap(True)
        self.summary_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary_text.setStyleSheet(f"""
            color: {P['ink_4']};
            font-family: '{FONT}';
            font-size: 18px;
            font-style: italic;
        """)
        self.summary_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.summary_text, stretch=1)

        lay.addSpacing(20)

        total_bg = QWidget()
        total_bg.setStyleSheet(f"""
            background: {P['o_xlight']};
            border-radius: {R};
            border: 1px solid {P['o_light']};
        """)
        tblay = QHBoxLayout(total_bg)
        tblay.setContentsMargins(18, 14, 18, 14)

        lbl_t = QLabel("รวมทั้งหมด")
        lbl_t.setStyleSheet(f"""
            color: {P['ink_3']};
            font-family: '{FONT}';
            font-size: 15px;
            font-weight: 500;
        """)

        num_row = QHBoxLayout()
        num_row.setSpacing(6)
        num_row.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.lbl_total = QLabel("—")
        self.lbl_total.setStyleSheet(f"""
            color: {P['o']};
            font-family: '{MONO}';
            font-size: 40px;
            font-weight: 700;
            line-height: 1;
        """)
        lbl_unit = QLabel("รายการ")
        lbl_unit.setStyleSheet(f"""
            color: {P['ink_3']};
            font-family: '{FONT}';
            font-size: 15px;
            font-weight: 500;
        """)
        lbl_unit.setAlignment(Qt.AlignBottom)
        num_row.addWidget(self.lbl_total)
        num_row.addWidget(lbl_unit, alignment=Qt.AlignBottom)

        tblay.addWidget(lbl_t, alignment=Qt.AlignVCenter)
        tblay.addLayout(num_row)
        lay.addWidget(total_bg)

        return panel

    # ───────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str = None):
        self.lbl_status.setText(text)
        self.status_dot.set_color(color or P['g_glow'])

    def _update_mode_btns(self):
        self.btn_pills.set_active(self.current_mode == "pills")
        self.btn_boxes.set_active(self.current_mode == "boxes")

    # ───────────────────────────────────────────────────────
    #   OPTIMIZED TICK LOOP — separated into 3 timers
    # ───────────────────────────────────────────────────────
    def _tick_video(self):
        """~30 fps: grab frame, push to queue, show immediately."""
        frame = self.camera.get_frame()
        if frame is None:
            return
        # Keep latest 2 frames for detect consumer
        self._frame_queue.append(frame)
        self._show_video(frame)

    def _tick_detect(self):
        """150 ms: submit detect job only if previous finished."""
        if self._detect_future is not None and not self._detect_future.done():
            return  # previous job still running — skip
        if not self._frame_queue:
            return
        frame = self._frame_queue[-1]  # grab newest
        self._detect_future = self._executor.submit(self._detect_worker, frame.copy())

    def _tick_ui(self):
        """50 ms: harvest result from background thread → update UI."""
        if self._detect_future is None or not self._detect_future.done():
            return
        try:
            inventory, images = self._detect_future.result()
        except Exception:
            return
        finally:
            self._detect_future = None

        # diff guard — skip full UI update if inventory unchanged
        if inventory == self._last_inventory and not images:
            return
        self._last_inventory = dict(inventory)

        self.tracker.update(
            {k: v for k, v in inventory.items()},
            {}  # bboxes already resolved inside worker
        )
        stable = self.tracker.get_stable()
        self._update_cards(stable, images)

    # ───────────────────────────────────────────────────────
    #   BACKGROUND WORKER  (runs in ThreadPoolExecutor)
    # ───────────────────────────────────────────────────────
    def _detect_worker(self, frame: np.ndarray):
        """Pure detection — no Qt calls allowed here."""
        h, w = frame.shape[:2]
        raw, imgs = {}, {}
        try:
            results = self.detector.model(frame, conf=CONF, verbose=False)
            for r in results:
                if not hasattr(r, 'boxes'):
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                    p = 10
                    crop = frame[max(0, y1-p):min(h, y2+p),
                                 max(0, x1-p):min(w, x2+p)]
                    if crop.size == 0:
                        continue
                    name, score = None, 0
                    if self.arcface_model and self.db_manager:
                        emb = self._embed_cached(crop)
                        if emb is not None:
                            name, score = self.db_manager.search(emb)
                    if name and score > CONF:
                        raw[name] = raw.get(name, 0) + 1
                        if name not in imgs:
                            imgs[name] = crop
        except Exception:
            pass
        return raw, imgs

    # ───────────────────────────────────────────────────────
    def _embed_cached(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Hash-keyed embedding cache to skip redundant inference."""
        # Fast perceptual hash: resize to 16×16, use mean bytes as key
        small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_NEAREST)
        key   = hashlib.md5(small.tobytes()).hexdigest()
        if key in self._embed_cache:
            return self._embed_cache[key]
        emb = self._embed(img)
        if emb is not None:
            if len(self._embed_cache) >= self._embed_cache_max:
                # evict oldest (pop arbitrary key)
                self._embed_cache.pop(next(iter(self._embed_cache)))
            self._embed_cache[key] = emb
        return emb

    # ───────────────────────────────────────────────────────
    def _update_cards(self, inventory: Dict[str, int], images: Dict[str, np.ndarray]):
        # Remove stale cards
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
                # Only update count label text if changed
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

        # Badge
        self.lbl_count.setText(str(len(inventory)) if inventory else "0")

        # Summary — only rebuild HTML when inventory changes
        if inventory:
            lines = []
            for name, count in sorted_items:
                sel = self.selected_name == name.lower()
                if sel:
                    row = (
                        f'<span style="color:{P["o"]};font-size:9px;">▶</span>'
                        f'&nbsp;<span style="color:{P["g"]};font-family:{FONT};'
                        f'font-size:20px;font-weight:700;">{name.upper()}</span>'
                        f'&nbsp;&nbsp;<span style="color:{P["o"]};font-family:{MONO};'
                        f'font-size:20px;font-weight:700;">{count}</span>'
                        f'<span style="color:{P["ink_3"]};font-family:{FONT};'
                        f'font-size:16px;"> รายการ</span>'
                    )
                else:
                    row = (
                        f'<span style="color:{P["ink_4"]};font-size:9px;">▸</span>'
                        f'&nbsp;<span style="color:{P["ink_2"]};font-family:{FONT};'
                        f'font-size:20px;font-weight:600;">{name.upper()}</span>'
                        f'&nbsp;&nbsp;<span style="color:{P["o_bright"]};font-family:{MONO};'
                        f'font-size:20px;font-weight:700;">{count}</span>'
                        f'<span style="color:{P["ink_4"]};font-family:{FONT};'
                        f'font-size:16px;"> รายการ</span>'
                    )
                lines.append(row)

            self.summary_text.setText("<br>".join(lines))
            self.summary_text.setStyleSheet(f"""
                color: {P['ink_2']};
                font-family: '{FONT}';
                font-size: 20px;
                font-style: normal;
                line-height: 2;
            """)
            self.lbl_total.setText(str(sum(inventory.values())))
        else:
            self.summary_text.setText("ยังไม่พบรายการ")
            self.summary_text.setStyleSheet(f"""
                color: {P['ink_4']};
                font-family: '{FONT}';
                font-size: 18px;
                font-style: italic;
            """)
            self.lbl_total.setText("—")

    # ───────────────────────────────────────────────────────
    def _on_card(self, name: str):
        nl = name.lower()
        if self.selected_name == nl:
            self.selected_name = None
            self._set_status("พร้อมใช้งาน", P['g_glow'])
            self.lbl_action_mode.setText(
                f"{'ยาเม็ด' if self.current_mode == 'pills' else 'กล่องยา'} · ระบบพร้อม"
            )
        else:
            self.selected_name = nl
            self._set_status(f"กำลังติดตาม  {name.upper()}", P['o_bright'])
            self.lbl_action_mode.setText(f"กำลังติดตาม:  {name.upper()}")
            self.lbl_action_mode.setStyleSheet(f"""
                color: {P['o']};
                font-family: '{FONT}';
                font-size: 14px;
                font-weight: 600;
            """)
        for cn, card in self.detection_cards.items():
            card.set_selected(self.selected_name == cn.lower())

    def clear_all_selections(self):
        self.selected_name = None
        self._set_status("พร้อมใช้งาน", P['g_glow'])
        mode_th = "ยาเม็ด" if self.current_mode == "pills" else "กล่องยา"
        self.lbl_action_mode.setText(f"{mode_th} · ระบบพร้อม")
        self.lbl_action_mode.setStyleSheet(f"""
            color: {P['ink_3']};
            font-family: '{FONT}';
            font-size: 14px;
        """)
        for card in self.detection_cards.values():
            card.set_selected(False)
    def _set_hardware_zoom(self, value: int):
        """ยิงคำสั่ง v4l2-ctl ไปที่ Driver กล้องโดยตรง"""
        try:
            # -d /dev/video0 คือ Default Camera Device
            cmd = ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl", f"zoom_absolute={value}"]
            subprocess.run(cmd, check=False, capture_output=True)
            # print(f"🔭 Hardware Zoom set to: {value}") 
        except Exception:
            pass # Silent fail (กรณีรันบน Windows หรือไม่มี Driver)
    def switch_mode(self, mode: str):
        self.current_mode = mode
        self.clear_all_selections()
        self._update_mode_btns()
        mode_th = "ยาเม็ด" if mode == "pills" else "กล่องยา"
        self.mode_tag.setText(f"MODE: {'PILLS' if mode == 'pills' else 'BOXES'}")
        self.lbl_action_mode.setText(f"{mode_th} · ระบบพร้อม")
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
        mode_th = "ยาเม็ด" if mode == "pills" else "กล่องยา"
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

        for base in ["models", "model_weights"]:
            p = os.path.join(BASE_DIR, base, "best.onnx")
            if os.path.exists(p):
                try: self.detector.load_model(p)
                except: pass
                break

        mname = "best_pill.pth" if mode == "pills" else "best_boxes.pth"
        mp = None
        for base in ["models", "model_weights"]:
            p = os.path.join(BASE_DIR, base, mname)
            if os.path.exists(p): mp = p; break

        if mp:
            try:
                model = PillModel(num_classes=1000, model_name='convnext_small', embed_dim=512)
                ckpt  = torch.load(mp, map_location=self.device, weights_only=True)
                clean = {k: v for k, v in ckpt.items() if not k.startswith('head')}
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
            emb = self.arcface_model(t).cpu().numpy().flatten()
            n   = np.linalg.norm(emb)
            return emb / n if n > 0 else emb
        except: return None

    def _show_video(self, frame: np.ndarray):
        try:
            h, w, c = frame.shape
            # Use Format_BGR888 directly — skip rgbSwapped() allocation
            qi = QImage(frame.data, w, h, c * w, QImage.Format_BGR888)
            self.lbl_video.setPixmap(
                QPixmap.fromImage(qi).scaled(
                    self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        except: pass

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


# ══════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Plus Jakarta Sans", 11))
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{
            font-family: 'Plus Jakarta Sans', 'DM Sans', system-ui;
        }}
        QToolTip {{
            background: {P['top_bg']};
            color: #CCCCCA;
            border: 1px solid {P['top_border']};
            padding: 6px 12px;
            border-radius: 8px;
            font-family: 'Plus Jakarta Sans';
            font-size: 12px;
        }}
    """)
    w = VisionStation()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()