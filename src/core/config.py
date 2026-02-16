import os

# หาตำแหน่ง Root Project อัตโนมัติ
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# การตั้งค่าระบบทั่วไป
SYSTEM_CONFIG = {
    "camera_index": 0,
    "window_size": (1400, 900),
    "fps": 30
}

# Paths สำหรับแต่ละโหมด (Master Version: รองรับทั้ง Admin, Doctor, Collector)
MODE_PATHS = {
    "pills": {
        # --- สำหรับ Collector (เก็บภาพดิบ) ---
        "raw_dir": os.path.join(BASE_DIR, "data", "raw_dataset", "raw_pills"),
        
        # --- สำหรับ Admin/Doctor (ใช้งานจริง) ---
        "db": os.path.join(BASE_DIR, "data", "pills", "hospital_pills.db"),
        "ref_img_dir": os.path.join(BASE_DIR, "data", "pills", "ref_images"),
        
        # --- AI Models ---
        "yolo_model": "yolo_pills.pt",       # ชื่อไฟล์ใน models/
        "rec_model": os.path.join(BASE_DIR, "models", "best_model_arcface.pth"), # 🔥 คีย์ที่หายไป ใส่กลับมาแล้ว!
        
        # --- Settings ---
        "use_rembg": True
    },
    "boxes": {
        # --- สำหรับ Collector ---
        "raw_dir": os.path.join(BASE_DIR, "data", "raw_dataset", "raw_boxes"),
        
        # --- สำหรับ Admin/Doctor ---
        "db": os.path.join(BASE_DIR, "data", "boxes", "hospital_boxes.db"),
        "ref_img_dir": os.path.join(BASE_DIR, "data", "boxes", "ref_images"),
        
        # --- AI Models ---
        "yolo_model": "yolo_boxes.pt",
        "rec_model": None,  # 🔥 กล่องยาไม่มี ArcFace ให้ใส่ None ไว้ (ห้ามลบคีย์ทิ้ง)
        
        # --- Settings ---
        "use_rembg": False
    }
}

# สร้างโฟลเดอร์ให้ครบกันเหนียว (Auto-Create)
for mode in MODE_PATHS:
    os.makedirs(MODE_PATHS[mode]["raw_dir"], exist_ok=True)
    os.makedirs(MODE_PATHS[mode]["ref_img_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(MODE_PATHS[mode]["db"]), exist_ok=True)