import os
import json

# ================= CONSTANTS =================
CONFIG_FILE = "presets.json"
DEFAULT_DATA_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "SnapSet_Data")

# Model Paths
ONNX_MODEL_PATH = os.path.join("models", "box_detector.onnx")
FALLBACK_MODEL = "yolov8n.pt"

# ================= HELPERS =================
def load_presets():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"Default": {"zoom": 1.0, "conf": 0.6}}

def save_presets(presets):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(presets, f, indent=4)