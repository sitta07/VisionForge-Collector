import os
from ultralytics import YOLO
from config import ONNX_MODEL_PATH, FALLBACK_MODEL

class ObjectDetector:
    def __init__(self):
        self.model = None
        self.model_name = "None"

    def load_model(self):
        path = ONNX_MODEL_PATH if os.path.exists(ONNX_MODEL_PATH) else FALLBACK_MODEL
        try:
            # 👉 แก้ตรงนี้: เพิ่ม task='detect' เพื่อปิด Warning
            self.model = YOLO(path, task='detect') 
            
            self.model_name = os.path.basename(path)
            return True, self.model_name
        except Exception as e:
            return False, str(e)

    def predict(self, frame, conf=0.5):
        if self.model:
            # verbose=False to keep terminal clean
            return self.model(frame, verbose=False, conf=conf)
        return []