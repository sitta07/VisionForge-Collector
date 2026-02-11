import torch
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 AI Device: {self.device.upper()}")
        self.model = None

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path, task='detect')
            self.model.to(self.device)
            return True, os.path.basename(model_path)
        except Exception as e:
            return False, str(e)

    def predict(self, frame, conf=0.5):
        if self.model is None: return None
        
        # Run YOLO
        results = self.model(frame, verbose=False, conf=conf, stream=False)
        
        if not results: return None

        # 🎯 Logic: หาตัวที่ Confidence สูงที่สุดแค่ตัวเดียว
        best_box = None
        highest_conf = -1

        for r in results:
            for box in r.boxes:
                current_conf = box.conf[0].item()
                if current_conf > highest_conf:
                    highest_conf = current_conf
                    best_box = box
        
        return best_box # คืนค่าเป็น Box Object ตัวเดียว
    
    def get_available_models(self, folder="models"):
        if not os.path.exists(folder): os.makedirs(folder)
        return [f for f in os.listdir(folder) if f.endswith(('.pt', '.onnx'))]