import os

# หาตำแหน่ง Root Project อัตโนมัติ (ไม่ว่าจะย้ายไปวางเครื่องไหนก็ทำงานได้)
# src/core/config.py -> src/core -> src -> VisionForge-System
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# การตั้งค่าระบบ
SYSTEM_CONFIG = {
    "camera_index": 0,
    "window_size": (1400, 900),
    "fps": 30
}

# Paths สำหรับแต่ละโหมด
MODE_PATHS = {
    "pills": {
        # เก็บภาพดิบ (สำหรับ Collector)
        "raw_dir": os.path.join(BASE_DIR, "data", "raw_dataset", "raw_pills"),
        # สำหรับ Admin/Doctor
        "db": os.path.join(BASE_DIR, "data", "pills", "hospital_pills.db"),
        "ref_img_dir": os.path.join(BASE_DIR, "data", "pills", "ref_images"),
        "yolo_model": "yolo_pills.pt" # ชื่อไฟล์ใน models/
    },
    "boxes": {
        # เก็บภาพดิบ (สำหรับ Collector)
        "raw_dir": os.path.join(BASE_DIR, "data", "raw_dataset", "raw_boxes"),
        # สำหรับ Admin/Doctor
        "db": os.path.join(BASE_DIR, "data", "boxes", "hospital_boxes.db"),
        "ref_img_dir": os.path.join(BASE_DIR, "data", "boxes", "ref_images"),
        "yolo_model": "yolo_boxes.pt"
    }
}

# สร้างโฟลเดอร์ให้ครบกันเหนียว
for mode in MODE_PATHS:
    os.makedirs(MODE_PATHS[mode]["raw_dir"], exist_ok=True)
    os.makedirs(MODE_PATHS[mode]["ref_img_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(MODE_PATHS[mode]["db"]), exist_ok=True)