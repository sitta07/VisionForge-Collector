import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import PIL.Image, PIL.ImageTk
import os
import time
import json
import threading
from ultralytics import YOLO
import numpy as np

# CONFIG
CONFIG_FILE = "presets.json"
DATA_ROOT = "data"
ONNX_MODEL_PATH = os.path.join("models", "box_detector.onnx") 
FALLBACK_MODEL = "yolov8n.pt"

class DataCollectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x750")
        
        # State
        self.is_recording = False
        self.camera_active = False
        self.model = None
        self.cap = None
        self.count_saved = 0
        
        # Default Settings
        self.zoom_level = 1.0
        self.presets = self.load_presets()

        # UI & Init
        self.create_ui()
        self.status_var.set("⏳ Loading Model...")
        self.window.update()
        self.load_model()
        self.start_camera()

    def create_ui(self):
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=0)
        self.window.rowconfigure(0, weight=1)

        # Video Frame
        self.video_frame = tk.Frame(self.window, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Controls
        self.controls_frame = tk.Frame(self.window, width=300, bg="#f0f0f0")
        self.controls_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)
        self.controls_frame.pack_propagate(False)

        lbl_style = {"font": ("Arial", 10, "bold"), "bg": "#f0f0f0"}
        
        # Section 1: Data
        tk.Label(self.controls_frame, text="📁 Class Name", **lbl_style).pack(pady=(10, 0), anchor="w")
        self.entry_name = tk.Entry(self.controls_frame, font=("Arial", 12))
        self.entry_name.pack(pady=5, fill="x", padx=10)
        self.entry_name.insert(0, "default_obj")
        tk.Button(self.controls_frame, text="📂 Open Folder", command=self.open_folder).pack(pady=5, fill="x", padx=10)

        ttk.Separator(self.controls_frame, orient='horizontal').pack(fill='x', pady=10)

        # Section 2: Presets
        tk.Label(self.controls_frame, text="⚙️ Presets", **lbl_style).pack(anchor="w")
        self.preset_combo = ttk.Combobox(self.controls_frame, values=list(self.presets.keys()))
        self.preset_combo.pack(pady=5, fill="x", padx=10)
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)

        btn_frame = tk.Frame(self.controls_frame, bg="#f0f0f0")
        btn_frame.pack(fill="x", padx=10)
        tk.Button(btn_frame, text="💾 Save", command=self.save_preset_dialog, width=10).pack(side="left", padx=2)
        tk.Button(btn_frame, text="❌ Del", command=self.delete_preset, width=10).pack(side="right", padx=2)

        ttk.Separator(self.controls_frame, orient='horizontal').pack(fill='x', pady=10)

        # Section 3: Camera
        tk.Label(self.controls_frame, text="🔍 Zoom Level", **lbl_style).pack(anchor="w")
        self.scale_zoom = tk.Scale(self.controls_frame, from_=1.0, to=3.0, resolution=0.1, orient="horizontal", command=self.update_zoom)
        self.scale_zoom.set(1.0)
        self.scale_zoom.pack(fill="x", padx=10)

        tk.Label(self.controls_frame, text="🎯 Confidence", **lbl_style).pack(anchor="w")
        self.scale_conf = tk.Scale(self.controls_frame, from_=0.1, to=1.0, resolution=0.05, orient="horizontal")
        self.scale_conf.set(0.6)
        self.scale_conf.pack(fill="x", padx=10)

        # Section 4: Actions
        self.btn_record = tk.Button(self.controls_frame, text="⏺ START RECORDING", bg="#ff4444", fg="white", font=("Arial", 12, "bold"), height=2, command=self.toggle_recording)
        self.btn_record.pack(fill="x", padx=10, pady=20)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.controls_frame, textvariable=self.status_var, bg="#ddd", height=2).pack(side="bottom", fill="x")

    def load_presets(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f: return json.load(f)
            except: pass
        return {"Default": {"zoom": 1.0, "conf": 0.6}}

    def save_presets_to_file(self):
        with open(CONFIG_FILE, 'w') as f: json.dump(self.presets, f)

    def save_preset_dialog(self):
        name = self.entry_name.get()
        if not name: return
        self.presets[f"Preset_{name}"] = {"zoom": self.scale_zoom.get(), "conf": self.scale_conf.get()}
        self.save_presets_to_file()
        self.preset_combo['values'] = list(self.presets.keys())
        messagebox.showinfo("Success", "Preset Saved")

    def delete_preset(self):
        name = self.preset_combo.get()
        if name in self.presets:
            del self.presets[name]
            self.save_presets_to_file()
            self.preset_combo['values'] = list(self.presets.keys())
            self.preset_combo.set("")

    def apply_preset(self, event):
        name = self.preset_combo.get()
        if name in self.presets:
            d = self.presets[name]
            self.scale_zoom.set(d.get("zoom", 1.0))
            self.scale_conf.set(d.get("conf", 0.6))
            self.zoom_level = d.get("zoom", 1.0)

    def load_model(self):
        # Logic: ลองโหลด ONNX ก่อน -> ถ้าไม่มีให้ใช้ YOLOv8n
        target_model = ""
        
        if os.path.exists(ONNX_MODEL_PATH):
            target_model = ONNX_MODEL_PATH
            self.status_var.set(f"⏳ Loading ONNX: {target_model}...")
        else:
            target_model = FALLBACK_MODEL
            self.status_var.set(f"⚠️ ONNX not found. Loading Default: {target_model}...")
        
        self.window.update() # บังคับให้หน้าจออัปเดตข้อความทันที

        try:
            # หมายเหตุ: ถ้าโมเดลเป็น Segmentation ให้เพิ่ม task='segment' ในวงเล็บ
            # เช่น: self.model = YOLO(target_model, task='segment')
            self.model = YOLO(target_model) 
            
            self.status_var.set(f"✅ Model Loaded: {os.path.basename(target_model)}")
            print(f"✅ Successfully loaded model from: {target_model}")
            
        except Exception as e:
            self.status_var.set(f"❌ Model Error: {e}")
            messagebox.showerror("Error", f"Could not load model: {e}")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.camera_active = True
        self.update_video_feed()

    def update_zoom(self, val):
        self.zoom_level = float(val)

    def apply_digital_zoom(self, frame):
        if self.zoom_level <= 1.0: return frame
        h, w = frame.shape[:2]
        nw, nh = int(w/self.zoom_level), int(h/self.zoom_level)
        x1, y1 = (w-nw)//2, (h-nh)//2
        return cv2.resize(frame[y1:y1+nh, x1:x1+nw], (w, h))

    def update_video_feed(self):
        if self.camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = self.apply_digital_zoom(frame)
                display = frame.copy()
                if self.is_recording and self.model:
                    self.process_and_save(frame, display)
                
                img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
                self.canvas.image = img
            self.window.after(10, self.update_video_feed)

    def process_and_save(self, raw, display):
        class_name = self.entry_name.get().strip()
        save_dir = os.path.join(DATA_ROOT, class_name)
        os.makedirs(save_dir, exist_ok=True)
        results = self.model(raw, verbose=False, conf=self.scale_conf.get())
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop = raw[y1:y2, x1:x2]
                if crop.size != 0:
                    cv2.imwrite(f"{save_dir}/{class_name}_{int(time.time()*1000)}.jpg", crop)
                    self.count_saved += 1
                    self.status_var.set(f"⏺ Recording... Saved: {self.count_saved}")

    def toggle_recording(self):
        if not self.is_recording:
            if not self.entry_name.get().strip(): return
            self.is_recording = True
            self.count_saved = 0
            self.btn_record.config(text="⏹ STOP", bg="black")
            self.entry_name.config(state="disabled")
        else:
            self.is_recording = False
            self.btn_record.config(text="⏺ RECORD", bg="#ff4444")
            self.entry_name.config(state="normal")
            self.status_var.set(f"✅ Saved {self.count_saved} images")

    def open_folder(self):
        p = os.path.abspath(DATA_ROOT)
        if not os.path.exists(p): os.makedirs(p)
        os.startfile(p) if os.name == 'nt' else os.system(f"open {p}")

    def on_closing(self):
        self.camera_active = False
        if self.cap: self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root, "SnapSet-Pro")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()