import cv2
import os

class CameraManager:
    def __init__(self):
        self.cap = None

    def start(self, idx=0):
        if self.cap: self.cap.release()
        # Windows DirectShow เพื่อความลื่นไหล
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(idx, backend)
        
        # Lock Resolution (ปรับได้ตามกล้อง)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        return self.cap.isOpened()

    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def trigger_autofocus(self):
        """ยิงคำสั่ง Autofocus (0->1)"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # OFF
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) # ON (Trigger)
            print("📸 Autofocus Triggered")

    def stop(self):
        if self.cap: self.cap.release()