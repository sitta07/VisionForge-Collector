import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([90, 255, 255])
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def apply_filters(self, frame, zoom=1.0, bright=0, contrast=1.0, preset="Default"):
        
        # 1. Preset (Texture)
        if preset == "Pill Enhanced (Texture)":
            frame = self._enhance_pill_texture(frame)
            
        # 2. Digital Zoom (High Quality Center Crop)
        if zoom > 1.0:
            h, w = frame.shape[:2]
            
            # คำนวณขนาดใหม่ตาม Zoom Factor
            new_w = int(w / zoom)
            new_h = int(h / zoom)
            
            # หาจุดกึ่งกลางเพื่อ Crop
            center_x = w // 2
            center_y = h // 2
            
            x1 = center_x - (new_w // 2)
            y1 = center_y - (new_h // 2)
            
            # กันค่าติดลบ
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = x1 + new_w
            y2 = y1 + new_h

            # Crop ภาพ
            cropped = frame[y1:y2, x1:x2]
            
            # Resize กลับมาเท่าขนาดเดิม (ใช้ LANCZOS4 เพื่อความคมชัดสูงสุด)
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # 3. Brightness/Contrast
        if bright != 0 or contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        
        return frame

    def _enhance_pill_texture(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def draw_crosshair(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        length, gap = 20, 5
        color = (0, 255, 0)
        
        cv2.line(frame, (cx - length, cy), (cx - gap, cy), color, 2)
        cv2.line(frame, (cx + gap, cy), (cx + length, cy), color, 2)
        cv2.line(frame, (cx, cy - length), (cx, cy - gap), color, 2)
        cv2.line(frame, (cx, cy + gap), (cx, cy + length), color, 2)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1) 
        return frame

    def remove_background(self, img):
            if img is None or img.size == 0: return img
            
            # 1. แปลงเป็น HSV และสร้าง Mask ของพื้นหลังสีเขียว
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bg_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            
            # 2. กลับค่า Mask (ให้ Object เป็นสีขาว, พื้นหลังเป็นดำ)
            # ตอนนี้ตรงโลโก้จะเป็นสีดำ (รูโหว่) เพราะมันเป็นสีเขียว
            object_mask = cv2.bitwise_not(bg_mask)
            
            # 3. หา Contours (เส้นขอบ) ของวัตถุ
            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return img # ถ้าหาไม่เจอ ให้คืนรูปเดิมไปก่อน
                
            # 4. หา Contour ที่ใหญ่ที่สุด (สมมติว่าเป็นกล่องยา)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 5. สร้าง Mask อันใหม่ที่สะอาดขึ้นมา
            clean_mask = np.zeros_like(bg_mask)
            
            # 6. วาด Contour ที่ใหญ่ที่สุดลงไป แล้ว "ระบายสีทับให้เต็ม (Fill)"
            # -1 คือการระบายสีทึบ สิ่งนี้จะช่วย "อุดรู" ตรงโลโก้ต้นไม้ครับ
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # 7. ตัดภาพด้วย Mask ใหม่ที่อุดรูแล้ว
            result = cv2.bitwise_and(img, img, mask=clean_mask)
            
            return result