import sys
import subprocess
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox, QCheckBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# --- ⚙️ CONFIG (Map ตาม Output ของคุณเป๊ะๆ) ---
CAMERA_DEVICE = "/dev/video0"

class LinuxCameraTuner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"VisionForge: Linux Camera Tuner ({CAMERA_DEVICE})")
        self.resize(1100, 750)
        
        # --- 1. Camera Feed ---
        self.cap = cv2.VideoCapture(0) # Index 0 usually maps to /dev/video0
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # --- UI Setup ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Video View
        self.lbl_video = QLabel("Loading Camera...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background: #000; border: 2px solid #555;")
        self.lbl_video.setMinimumSize(640, 480)
        self.layout.addWidget(self.lbl_video, stretch=3)

        # Controls
        self.panel = QWidget()
        self.panel.setStyleSheet("background: #222; border-left: 1px solid #444;")
        self.ctrl_layout = QVBoxLayout(self.panel)
        self.ctrl_layout.setSpacing(15)
        
        lbl_title = QLabel("🎛️ V4L2 CONTROLS")
        lbl_title.setStyleSheet("color: #0f0; font-weight: bold; font-size: 16px;")
        lbl_title.setAlignment(Qt.AlignCenter)
        self.ctrl_layout.addWidget(lbl_title)

        # --- 🎚️ Sliders (Mapping ตาม v4l2-ctl ของคุณ) ---
        
        # 1. Zoom (min=50 max=98 default=50)
        self.add_slider("Zoom (Digital)", "zoom_absolute", 50, 98, 50)

        # 2. Sharpness (min=0 max=20 default=10)
        self.add_slider("Sharpness", "sharpness", 0, 20, 10)

        # 3. Contrast (min=0 max=100 default=50)
        self.add_slider("Contrast", "contrast", 0, 100, 50)

        # 4. Saturation (min=0 max=100 default=50)
        self.add_slider("Saturation", "saturation", 0, 100, 50)

        # 5. White Balance Temp (min=2000 max=10000 default=4150)
        # ต้องปิด Auto WB ก่อนถึงจะปรับได้
        self.chk_wb_auto = QCheckBox("Auto White Balance")
        self.chk_wb_auto.setChecked(True)
        self.chk_wb_auto.stateChanged.connect(self.toggle_wb_auto)
        self.ctrl_layout.addWidget(self.chk_wb_auto)

        self.wb_slider = self.add_slider("WB Temperature", "white_balance_temperature", 2000, 10000, 4150)
        self.wb_slider.setEnabled(False) # Disable at start if Auto is On

        # Reset Button
        self.ctrl_layout.addStretch()
        btn_reset = QPushButton("🔄 RESET DEFAULTS")
        btn_reset.setFixedHeight(45)
        btn_reset.setStyleSheet("background-color: #c0392b; font-weight: bold;")
        btn_reset.clicked.connect(self.reset_defaults)
        self.ctrl_layout.addWidget(btn_reset)

        self.layout.addWidget(self.panel, stretch=1)

        # Timer loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.apply_styles()

    def add_slider(self, label, v4l2_name, min_val, max_val, default):
        grp = QGroupBox(label)
        v = QVBoxLayout(grp)
        
        h = QHBoxLayout()
        lbl_val = QLabel(str(default))
        lbl_val.setStyleSheet("color: #0ff; font-weight: bold;")
        h.addWidget(QLabel("Val:"))
        h.addWidget(lbl_val)
        v.addLayout(h)
        
        sl = QSlider(Qt.Horizontal)
        sl.setRange(min_val, max_val)
        sl.setValue(default)
        # ส่งค่าไปฟังก์ชัน set_v4l2 เมื่อ slider ขยับ
        sl.valueChanged.connect(lambda val, n=v4l2_name, l=lbl_val: self.set_v4l2(n, val, l))
        
        v.addWidget(sl)
        self.ctrl_layout.addWidget(grp)
        return sl

    def set_v4l2(self, name, value, label_widget=None):
        """🔥 The Magic: Calls linux terminal command directly"""
        if label_widget:
            label_widget.setText(str(value))
        
        cmd = ["v4l2-ctl", "-d", CAMERA_DEVICE, "--set-ctrl", f"{name}={value}"]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Executed: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error setting {name}: {e}")

    def toggle_wb_auto(self, state):
        is_auto = 1 if state == 2 else 0 # 2=Checked, 0=Unchecked
        self.set_v4l2("white_balance_automatic", is_auto)
        
        # Enable/Disable manual slider
        self.wb_slider.setEnabled(not is_auto)

    def reset_defaults(self):
        # Reset to defaults from your log
        self.set_v4l2("zoom_absolute", 50)
        self.set_v4l2("sharpness", 10)
        self.set_v4l2("contrast", 50)
        self.set_v4l2("saturation", 50)
        self.set_v4l2("white_balance_automatic", 1)
        self.chk_wb_auto.setChecked(True)
        # Update UI sliders manually if needed (skipped for brevity)
        print("🔄 Settings Reset!")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # frame = cv2.flip(frame, 1) # Mirror if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame.shape
            qi = QImage(frame.data, w, h, c*w, QImage.Format_RGB888)
            self.lbl_video.setPixmap(QPixmap.fromImage(qi).scaled(
                self.lbl_video.size(), Qt.KeepAspectRatio))

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; color: #ddd; font-family: sans-serif; }
            QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; background: #1a1a1a; }
            QCheckBox { color: #fff; spacing: 5px; }
            QCheckBox::indicator { width: 15px; height: 15px; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = LinuxCameraTuner()
    w.show()
    sys.exit(app.exec())