import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import PIL.Image, PIL.ImageTk
import os
import time

# Import Modules
import config
from camera import CameraManager
from detector import ObjectDetector

class DataCollectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1280x800")
        
        # --- Modules ---
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        
        # --- State ---
        self.is_recording = False
        self.save_root_dir = config.DEFAULT_DATA_ROOT
        self.count_saved = 0
        self.zoom_level = 1.0
        self.presets = config.load_presets()

        # --- Init UI & Backend ---
        self.create_ui()
        self.init_backend()

    def init_backend(self):
        # Load AI
        self.status_var.set("⏳ Loading Model...")
        self.window.update()
        success, msg = self.detector.load_model()
        if success:
            self.status_var.set(f"✅ Model: {msg}")
        else:
            self.status_var.set(f"❌ Error: {msg}")
        
        # Start Camera
        self.start_camera_stream(0)

    def start_camera_stream(self, idx):
        if self.camera.start(idx):
            self.update_video_feed()
        else:
            messagebox.showerror("Error", f"Cannot open Camera {idx}")

    def create_ui(self):
        # Grid Setup
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=0)
        self.window.rowconfigure(0, weight=1)

        # Left: Video
        self.video_frame = tk.Frame(self.window, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Right: Controls
        self.controls_frame = tk.Frame(self.window, width=320, bg="#f5f5f5")
        self.controls_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)
        self.controls_frame.pack_propagate(False)

        self._build_controls()

    def _build_controls(self):
        lbl_style = {"font": ("Segoe UI", 10, "bold"), "bg": "#f5f5f5"}
        
        # 1. Output Path
        tk.Label(self.controls_frame, text="📂 Output Folder", **lbl_style).pack(anchor="w", pady=(10,0))
        self.lbl_path = tk.Label(self.controls_frame, text=self.save_root_dir[-40:], fg="gray", bg="#f5f5f5")
        self.lbl_path.pack(anchor="w", padx=10)
        tk.Button(self.controls_frame, text="Change Folder...", command=self.change_folder).pack(fill="x", padx=10, pady=5)
        
        ttk.Separator(self.controls_frame, orient='horizontal').pack(fill='x', pady=10)

        # 2. Camera
        tk.Label(self.controls_frame, text="📹 Camera Source", **lbl_style).pack(anchor="w")
        cam_frame = tk.Frame(self.controls_frame, bg="#f5f5f5")
        cam_frame.pack(fill="x", padx=10)
        self.cam_combo = ttk.Combobox(cam_frame, values=["Cam 0", "Cam 1", "Cam 2"], width=10)
        self.cam_combo.current(0)
        self.cam_combo.pack(side="left", fill="x", expand=True)
        tk.Button(cam_frame, text="🔄 Reload", command=self.reload_camera).pack(side="right", padx=5)

        ttk.Separator(self.controls_frame, orient='horizontal').pack(fill='x', pady=10)

        # 3. Class & Preset
        tk.Label(self.controls_frame, text="🏷️ Class Name", **lbl_style).pack(anchor="w")
        self.entry_name = tk.Entry(self.controls_frame, font=("Segoe UI", 11))
        self.entry_name.pack(fill="x", padx=10, pady=5)
        self.entry_name.insert(0, "default_obj")

        tk.Label(self.controls_frame, text="⚙️ Presets", **lbl_style).pack(anchor="w", pady=(10,0))
        self.preset_combo = ttk.Combobox(self.controls_frame, values=list(self.presets.keys()))
        self.preset_combo.pack(fill="x", padx=10, pady=5)
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)

        btn_preset = tk.Frame(self.controls_frame, bg="#f5f5f5")
        btn_preset.pack(fill="x", padx=10)
        tk.Button(btn_preset, text="Save Preset", command=self.save_preset_ui).pack(side="left", expand=True, fill="x")
        tk.Button(btn_preset, text="Del Preset", command=self.delete_preset_ui).pack(side="right", expand=True, fill="x")

        # 4. Zoom & Conf
        tk.Label(self.controls_frame, text="🔍 Zoom", **lbl_style).pack(anchor="w", pady=(15,0))
        self.scale_zoom = tk.Scale(self.controls_frame, from_=1.0, to=3.0, resolution=0.1, orient="horizontal", command=self.set_zoom)
        self.scale_zoom.set(1.0)
        self.scale_zoom.pack(fill="x", padx=10)

        tk.Label(self.controls_frame, text="🎯 Confidence", **lbl_style).pack(anchor="w")
        self.scale_conf = tk.Scale(self.controls_frame, from_=0.1, to=1.0, resolution=0.05, orient="horizontal")
        self.scale_conf.set(0.6)
        self.scale_conf.pack(fill="x", padx=10)

        # 5. Record
        self.btn_record = tk.Button(self.controls_frame, text="⏺ START RECORDING", bg="#ff4444", fg="white", font=("Segoe UI", 12, "bold"), height=2, command=self.toggle_record)
        self.btn_record.pack(fill="x", padx=10, pady=20)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.controls_frame, textvariable=self.status_var, bg="#ddd", height=2).pack(side="bottom", fill="x")

    # --- Logic Connectors ---
    def update_video_feed(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # 1. Digital Zoom
            frame = self._apply_zoom(frame)
            display = frame.copy()
            
            # 2. AI & Record
            if self.is_recording:
                self._process_recording(frame, display)
            
            # 3. Render to Canvas
            img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
            self.canvas.image = img # Keep ref
            
        self.window.after(10, self.update_video_feed)

    def _apply_zoom(self, frame):
        if self.zoom_level <= 1.0: return frame
        h, w = frame.shape[:2]
        nw, nh = int(w/self.zoom_level), int(h/self.zoom_level)
        x1, y1 = (w-nw)//2, (h-nh)//2
        return cv2.resize(frame[y1:y1+nh, x1:x1+nw], (w, h))

    def _process_recording(self, raw, display):
        class_name = self.entry_name.get().strip()
        save_dir = os.path.join(self.save_root_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Predict
        results = self.detector.predict(raw, conf=self.scale_conf.get())
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Save Crop
                crop = raw[y1:y2, x1:x2]
                if crop.size != 0:
                    fname = f"{class_name}_{int(time.time()*1000)}.jpg"
                    cv2.imwrite(os.path.join(save_dir, fname), crop)
                    self.count_saved += 1
                    self.status_var.set(f"⏺ Saved: {self.count_saved} | {fname}")

    # --- Actions ---
    def toggle_record(self):
        if not self.is_recording:
            if not self.entry_name.get().strip():
                messagebox.showwarning("Warning", "Enter Class Name first!")
                return
            self.is_recording = True
            self.count_saved = 0
            self.btn_record.config(text="⏹ STOP", bg="black")
            self.entry_name.config(state="disabled")
        else:
            self.is_recording = False
            self.btn_record.config(text="⏺ RECORD", bg="#ff4444")
            self.entry_name.config(state="normal")
            messagebox.showinfo("Done", f"Saved {self.count_saved} images.")

    def reload_camera(self):
        idx = int(self.cam_combo.get().split(" ")[1])
        self.start_camera_stream(idx)

    def change_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.save_root_dir = p
            self.lbl_path.config(text=p[-40:])

    def set_zoom(self, val): self.zoom_level = float(val)

    # --- Presets (Using Config Module) ---
    def save_preset_ui(self):
        name = self.entry_name.get()
        if not name: return
        self.presets[f"Preset_{name}"] = {"zoom": self.scale_zoom.get(), "conf": self.scale_conf.get()}
        config.save_presets(self.presets)
        self.preset_combo['values'] = list(self.presets.keys())
        messagebox.showinfo("Saved", f"Preset_{name}")

    def delete_preset_ui(self):
        name = self.preset_combo.get()
        if name in self.presets:
            del self.presets[name]
            config.save_presets(self.presets)
            self.preset_combo['values'] = list(self.presets.keys())
            self.preset_combo.set("")

    def apply_preset(self, event):
        name = self.preset_combo.get()
        if name in self.presets:
            d = self.presets[name]
            self.scale_zoom.set(d.get("zoom", 1.0))
            self.scale_conf.set(d.get("conf", 0.6))
            self.zoom_level = d.get("zoom", 1.0)

    def on_closing(self):
        """Clean up resources before closing"""
        if self.camera:
            self.camera.stop()
        self.window.destroy()
        print("👋 App Closed Successfully")