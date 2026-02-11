import cv2
import os

class CameraManager:
    def __init__(self):
        self.cap = None
        self.active = False
        
    def start(self, index=0):
        self.stop() # Ensure old stream is closed
        
        # 🔧 FIX: DirectShow for Windows (Fixes YoloLiv Freeze)
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce Lag
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if self.cap.isOpened():
            self.active = True
            return True
        return False

    def get_frame(self):
        if self.active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def stop(self):
        self.active = False
        if self.cap:
            self.cap.release()
            self.cap = None