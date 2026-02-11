import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        # ค่าสีเขียว (Green Screen)
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([90, 255, 255])

    def apply_filters(self, frame, zoom=1.0, bright=0, contrast=1.0):
        # 1. Digital Zoom
        if zoom > 1.0:
            h, w = frame.shape[:2]
            nw, nh = int(w/zoom), int(h/zoom)
            x1, y1 = (w-nw)//2, (h-nh)//2
            frame = cv2.resize(frame[y1:y1+nh, x1:x1+nw], (w, h))

        # 2. Brightness/Contrast
        # clip เพื่อไม่ให้ค่าสีเพี้ยนเวลาปรับเยอะๆ
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        return frame

    def draw_crosshair(self, frame):
        """วาดเป้าเล็งกลางจอ (Crosshair)"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Style: เป้าสีเขียว Neon แบบเกม FPS
        length = 20
        gap = 5
        color = (0, 255, 0) # Green
        thickness = 2
        
        # เส้นแนวนอน
        cv2.line(frame, (cx - length, cy), (cx - gap, cy), color, thickness)
        cv2.line(frame, (cx + gap, cy), (cx + length, cy), color, thickness)
        # เส้นแนวตั้ง
        cv2.line(frame, (cx, cy - length), (cx, cy - gap), color, thickness)
        cv2.line(frame, (cx, cy + gap), (cx, cy + length), color, thickness)
        
        # จุดแดงตรงกลางเป๊ะๆ
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1) 
        
        return frame

    def remove_background(self, img):
        """เจาะสีเขียวออก (Chromakey)"""
        if img is None or img.size == 0: return img
        
        # แปลงเป็น HSV เพื่อจับสีเขียวได้ดีกว่า
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # กลับ Mask (เอาส่วนที่ไม่ใช่สีเขียว)
        mask_inv = cv2.bitwise_not(mask)
        
        # ตัดพื้นหลังออก
        fg = cv2.bitwise_and(img, img, mask=mask_inv)
        return fg