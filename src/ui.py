import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import PIL.Image, PIL.ImageTk
import os
import time

# Modules
import config
from camera import CameraManager
from detector import ObjectDetector
from image_utils import ImageProcessor

# Setup Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class DataCollectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisionForge: Smart Collector")
        self.geometry("1400x850")
        
        # --- Backend ---
        self.camera = CameraManager()
        self.detector = ObjectDetector()
        self.processor = ImageProcessor()
        self.presets = config.load_presets()
        
        # --- State ---
        self.is_recording = False
        self.save_dir = config.DEFAULT_DATA_ROOT
        self.count_saved = 0
        self.frame_count = 0
        
        # --- UI Layout ---
        self.grid_columnconfigure(0, weight=3) # Video
        self.grid_columnconfigure(1, weight=1) # Controls
        self.grid_rowconfigure(0, weight=1)

        self.setup_ui()
        
        # --- Start ---
        self.refresh_models()
        self.camera.start(0)
        self.update_loop()

    def setup_ui(self):
        # 1. Video Panel (Left)
        self.video_frame = ctk.CTkFrame(self, fg_color="#050505", corner_radius=0)
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.lbl_video = ctk.CTkLabel(self.video_frame, text="Loading Camera...", font=("Arial", 16))
        self.lbl_video.pack(fill="both", expand=True)

        # 2. Sidebar (Right)
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#1a1a1a")
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.pack_propagate(False)

        # --- Top: Preview ---
        self.preview_box = ctk.CTkFrame(self.sidebar, height=200, fg_color="#2b2b2b")
        self.preview_box.pack(fill="x", padx=10, pady=10)
        self.preview_box.pack_propagate(False)
        ctk.CTkLabel(self.preview_box, text="LIVE PREVIEW (SAVE OUTPUT)", font=("Arial", 10, "bold")).pack(pady=2)
        self.lbl_preview = ctk.CTkLabel(self.preview_box, text="WAITING...", text_color="gray")
        self.lbl_preview.pack(expand=True)

        # --- Scrollable Controls ---
        self.scroll = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Group: Model & Cam
        self.add_header("🧠 SYSTEM")
        self.combo_model = ctk.CTkComboBox(self.scroll, values=[], command=self.load_selected_model)
        self.combo_model.pack(fill="x", pady=5)
        ctk.CTkButton(self.scroll, text="⚡ TRIGGER AUTOFOCUS", command=self.camera.trigger_autofocus, fg_color="#333").pack(fill="x", pady=2)

        # Group: Image & Presets
        self.add_header("🎨 IMAGE & PRESETS")
        
        # Preset Controls
        row_preset = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row_preset.pack(fill="x", pady=2)
        self.combo_preset = ctk.CTkComboBox(row_preset, values=list(self.presets.keys()), command=self.apply_preset)
        self.combo_preset.pack(side="left", fill="x", expand=True, padx=(0,5))
        ctk.CTkButton(row_preset, text="💾", width=40, command=self.save_current_preset).pack(side="right")
        ctk.CTkButton(row_preset, text="🗑", width=40, fg_color="#c0392b", command=self.delete_preset).pack(side="right", padx=2)

        # Sliders
        self.sl_zoom = self.add_slider("Zoom", 1.0, 4.0, 1.0)
        self.sl_bright = self.add_slider("Brightness", -100, 100, 0)
        self.sl_cont = self.add_slider("Contrast", 0.5, 3.0, 1.0)

        # Group: Recording
        self.add_header("⏺ RECORDING SETUP")
        self.entry_class = ctk.CTkEntry(self.scroll, placeholder_text="Class Name")
        self.entry_class.pack(fill="x", pady=5)
        
        # Folder
        row_path = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row_path.pack(fill="x", pady=5)
        self.btn_path = ctk.CTkButton(row_path, text="📂 FOLDER", width=80, command=self.change_folder)
        self.btn_path.pack(side="left")
        self.lbl_path = ctk.CTkLabel(row_path, text=".../Desktop", font=("Arial", 10))
        self.lbl_path.pack(side="right", padx=5)

        # Interval Slider
        self.sl_interval = self.add_slider("Save Every N Frames", 1, 60, 5) # Default 5
        self.sl_conf = self.add_slider("Confidence Threshold", 0.1, 1.0, 0.6)

        # --- Bottom: Record Button ---
        self.btn_record = ctk.CTkButton(self.sidebar, text="START RECORDING", height=50, 
                                        fg_color="#e74c3c", hover_color="#c0392b", 
                                        font=("Arial", 16, "bold"), command=self.toggle_record)
        self.btn_record.pack(fill="x", padx=10, pady=20, side="bottom")

    def add_header(self, text):
        ctk.CTkLabel(self.scroll, text=text, anchor="w", font=("Arial", 12, "bold"), text_color="#3498db").pack(fill="x", pady=(15, 2))

    def add_slider(self, label, min_v, max_v, default):
        ctk.CTkLabel(self.scroll, text=label, anchor="w", font=("Arial", 11)).pack(fill="x")
        sl = ctk.CTkSlider(self.scroll, from_=min_v, to=max_v)
        sl.set(default)
        sl.pack(fill="x", pady=(0, 10))
        return sl

    # --- Logic ---
    def update_loop(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # 1. Image FX (Zoom/Bright/Cont)
            zoom = self.sl_zoom.get()
            bright = self.sl_bright.get()
            cont = self.sl_cont.get()
            
            processed = self.processor.apply_filters(frame, zoom=zoom, bright=bright, contrast=cont)
            display = processed.copy()
            
            # Draw Crosshair
            self.processor.draw_crosshair(display)

            # 2. AI Detect
            conf_thresh = self.sl_conf.get()
            best_box = self.detector.predict(processed, conf=conf_thresh)
            
            latest_crop = None

            if best_box is not None: # ต้องเช็ค is not None เสมอ
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                conf = best_box.conf[0].item()
                cls_id = int(best_box.cls[0]) # ดึง ID ออกมาก่อน
                
                # 🔥 FIX CRASH: ใช้ .get() เพื่อกัน error ถ้าหาชื่อไม่เจอ
                class_name = self.detector.model.names.get(cls_id, f"ID_{cls_id}")
                label = f"{class_name} {conf:.2f}"
                
                # Draw Box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Crop Logic
                crop = processed[y1:y2, x1:x2]
                if crop.size > 0:
                    latest_crop = self.processor.remove_background(crop)
                    
                    # Save Logic
                    if self.is_recording:
                        self.frame_count += 1
                        interval = int(self.sl_interval.get())
                        if self.frame_count % interval == 0:
                            self.save_image(latest_crop)

            # 3. UI Update
            self.show_video(display)
            self.show_preview(latest_crop)

        self.after(20, self.update_loop)

    def show_video(self, img):
        # Resize to fit window height logic (Basic)
        h, w = img.shape[:2]
        win_h = self.video_frame.winfo_height()
        if win_h > 100:
            scale = win_h / h
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ctk_img = ctk.CTkImage(PIL.Image.fromarray(img_rgb), size=(img.shape[1], img.shape[0]))
        self.lbl_video.configure(image=ctk_img, text="")

    def show_preview(self, img):
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ctk_img = ctk.CTkImage(PIL.Image.fromarray(img_rgb), size=(180, 180)) # Fixed Box
        self.lbl_preview.configure(image=ctk_img, text="")

    def save_image(self, img):
        cname = self.entry_class.get().strip() or "unknown"
        path = os.path.join(self.save_dir, cname)
        os.makedirs(path, exist_ok=True)
        
        fname = f"{int(time.time()*1000)}.png"
        cv2.imwrite(os.path.join(path, fname), img)
        self.count_saved += 1
        self.btn_record.configure(text=f"REC • {self.count_saved}")

    # --- Actions ---
    def toggle_record(self):
        if not self.is_recording:
            if not self.entry_class.get().strip():
                messagebox.showwarning("Warning", "Enter Class Name first!")
                return
            self.is_recording = True
            self.count_saved = 0
            self.btn_record.configure(text="STOP", fg_color="#333")
            self.entry_class.configure(state="disabled")
            self.sl_interval.configure(state="disabled") # Lock interval while recording
        else:
            self.is_recording = False
            self.btn_record.configure(text="START RECORDING", fg_color="#e74c3c")
            self.entry_class.configure(state="normal")
            self.sl_interval.configure(state="normal")
            messagebox.showinfo("Done", f"Saved {self.count_saved} images.")

    def refresh_models(self):
        models = self.detector.get_available_models()
        if models:
            self.combo_model.configure(values=models)
            self.combo_model.set(models[0])
            self.load_selected_model(models[0])

    def load_selected_model(self, name):
        path = os.path.join("models", name)
        success, msg = self.detector.load_model(path)
        print(f"Model: {msg}")

    def change_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.save_dir = p
            self.lbl_path.configure(text=f".../{os.path.basename(p)}")

    # --- Preset Logic ---
    def save_current_preset(self):
        name = simpledialog.askstring("Save Preset", "Enter Preset Name:")
        if name:
            self.presets[name] = {
                "zoom": self.sl_zoom.get(),
                "bright": self.sl_bright.get(),
                "cont": self.sl_cont.get(),
                "interval": self.sl_interval.get(),
                "conf": self.sl_conf.get()
            }
            config.save_presets(self.presets)
            self.combo_preset.configure(values=list(self.presets.keys()))
            self.combo_preset.set(name)

    def apply_preset(self, name):
        if name in self.presets:
            d = self.presets[name]
            self.sl_zoom.set(d.get("zoom", 1.0))
            self.sl_bright.set(d.get("bright", 0))
            self.sl_cont.set(d.get("cont", 1.0))
            self.sl_interval.set(d.get("interval", 5))
            self.sl_conf.set(d.get("conf", 0.6))

    def delete_preset(self):
        name = self.combo_preset.get()
        if name in self.presets:
            if messagebox.askyesno("Confirm", f"Delete {name}?"):
                del self.presets[name]
                config.save_presets(self.presets)
                self.combo_preset.configure(values=list(self.presets.keys()))
                self.combo_preset.set("")