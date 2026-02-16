import os
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                return False, f"Model not found: {model_path}"
            
            # 🔥 Senior Tip: ระบุ task='detect' ให้ชัดเจนเพื่อข้ามขั้นตอนเดาของระบบ
            self.model = YOLO(model_path, task='detect')
            return True, f"Loaded {model_path} successfully"
        except Exception as e:
            return False, str(e)

    def predict(self, frame, conf=0.5):
        if self.model is None:
            return None
        
        # 🔥 ระบุ device=0 เพื่อใช้ CUDA และสกัดเอาข้อมูลกล่อง (boxes) ออกมา
        results = self.model.predict(
            source=frame, 
            conf=conf, 
            verbose=False, 
            device=0  # 0 สำหรับ GPU, 'cpu' สำหรับ CPU
        )
        
        if results and len(results[0].boxes) > 0:
            return results[0].boxes[0]
        return None

    def get_available_models(self):
        if not os.path.exists("models"):
            return []
        return [f for f in os.listdir("models") if f.endswith(('.pt', '.onnx'))]