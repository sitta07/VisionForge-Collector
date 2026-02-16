import cv2
import os
import time

class CameraManager:
    def __init__(self):
        self.cap = None
        self.last_zoom_val = -1 
    
    def start(self, idx=0):
        if self.cap: 
            self.cap.release()
        
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(idx, backend)
        
        # บังคับ Full HD
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if self.cap.isOpened():
            # สั่ง Reset Zoom ไปที่ -100 ทันทีที่เปิด (Wide สุด)
            try:
                self.cap.set(cv2.CAP_PROP_ZOOM, -100)
                print("🔄 Zoom Init: -100 (Wide)")
            except:
                pass
            
            time.sleep(0.5)
            self.trigger_autofocus()
            return True
        return False
    
    def set_zoom(self, value):
        """
        Hardware Zoom
        value: 1.0 (Wide / No Zoom) -> 4.0 (Max Zoom)
        
        แปลงเป็น:
        1.0 -> -100 (Wide สุด, ไม่ซูม)
        1.5 -> -50
        2.0 -> 0
        3.0 -> 100
        4.0 -> 200 (Zoom สุด)
        """
        if not self.cap: 
            return
        
        # 🔥 FORCE RESET: ถ้าเป็น 1.0 ให้บังคับ reset เป็น -100
        if abs(value - 1.0) < 0.01:
            try:
                self.cap.set(cv2.CAP_PROP_ZOOM, -100)
                print(f"🔍 Force Reset Zoom to -100 (Wide)")
                self.last_zoom_val = 1.0
                return
            except Exception as e:
                print(f"❌ Zoom Error: {e}")
                return
        
        # เช็คกันคำสั่งรัว (เฉพาะค่าที่ไม่ใช่ 1.0)
        if abs(value - self.last_zoom_val) < 0.05: 
            return
        
        self.last_zoom_val = value
        
        # Linear mapping: 1.0->(-100) to 4.0->(200)
        driver_value = int(100 * (value - 1.0) - 100)
        
        # Clamp to valid range
        driver_value = max(-100, min(200, driver_value))
        
        try:
            self.cap.set(cv2.CAP_PROP_ZOOM, driver_value)
            print(f"🔍 Set Hardware Zoom: slider={value:.2f}x -> driver={driver_value}")
        except Exception as e:
            print(f"❌ Zoom Error: {e}")
    
    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None
    
    def trigger_autofocus(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    def stop(self):
        if self.cap: 
            self.cap.release()