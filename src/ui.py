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

# --- THEME CONFIG ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Colors
COLOR_BG = "#0f0f0f"        # พื้นหลังดำสนิท
COLOR_SIDEBAR = "#1e1e1e"   # แถบข้างสีเทาเข้ม
COLOR_CARD = "#2b2b2b"      # การ์ดรองปุ่ม
COLOR_ACCENT = "#3b8ed0"    # สีฟ้าหลัก
COLOR_RECORD = "#e74c3c"    # สีแดงอัด
FONT_HEADER = ("Roboto Medium", 14)
FONT_LABEL = ("Roboto", 11)
FONT_VALUE = ("Consolas", 10)

class DataCollectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VisionForge: Smart Collector Pro")
        self.geometry("1400x900")
        self.configure(fg_color=COLOR_BG)
        
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
        
        # --- Layout ---
        self.grid_columnconfigure(0, weight=1) # Video Area (ยืดได้)
        self.grid_columnconfigure(1, weight=0) # Sidebar (Fix)
        self.grid_rowconfigure(0, weight=1)

        self.setup_ui()
        
        # --- Start ---
        self.refresh_models()
        self.camera.start(0)
        self.update_loop()

    def setup_ui(self):
        # ================== 1. VIDEO AREA (LEFT) ==================
        self.video_frame = ctk.CTkFrame(self, fg_color="black", corner_radius=0)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        self.lbl_video = ctk.CTkLabel(self.video_frame, text="NO SIGNAL", font=("Roboto", 20))
        self.lbl_video.place(relx=0.5, rely=0.5, anchor="center")

        # Overlay Info (มุมซ้ายบนของวิดีโอ)
        self.info_overlay = ctk.CTkLabel(self.video_frame, text="READY", 
                                       fg_color="#111111", corner_radius=5, 
                                       font=("Consolas", 12), text_color="white")
        self.info_overlay.place(x=20, y=20)

        # ================== 2. SIDEBAR (RIGHT) ==================
        self.sidebar = ctk.CTkFrame(self, width=380, fg_color=COLOR_SIDEBAR, corner_radius=0)
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.grid_propagate(False) # ล็อกความกว้าง

        # --- HEADER ---
        title_lbl = ctk.CTkLabel(self.sidebar, text="VISION FORGE", font=("Impact", 24), text_color="#ffffff")
        title_lbl.pack(pady=(20, 5))
        ver_lbl = ctk.CTkLabel(self.sidebar, text="DATASET COLLECTOR v2.0", font=("Arial", 10), text_color="gray")
        ver_lbl.pack(pady=(0, 15))

        # --- SCROLLABLE CONTENT ---
        self.scroll = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=5)

        # === CARD 1: PREVIEW ===
        self.create_preview_card()

        # === CARD 2: SYSTEM & AI ===
        self.create_system_card()

        # === CARD 3: IMAGE ADJUSTMENT ===
        self.create_image_card()

        # === CARD 4: RECORDING ===
        self.create_recording_card()

        # --- FOOTER BUTTON ---
        self.btn_record = ctk.CTkButton(self.sidebar, text="START RECORDING", height=60,
                                        font=("Roboto", 16, "bold"),
                                        fg_color=COLOR_RECORD, hover_color="#c0392b",
                                        command=self.toggle_record)
        self.btn_record.pack(fill="x", padx=15, pady=20, side="bottom")

    # ================= UI HELPERS (CARD BUILDERS) =================
    def create_card_frame(self, title):
        card = ctk.CTkFrame(self.scroll, fg_color=COLOR_CARD, corner_radius=10)
        card.pack(fill="x", pady=8, ipady=5)
        ctk.CTkLabel(card, text=title, font=FONT_HEADER, text_color=COLOR_ACCENT).pack(anchor="w", padx=15, pady=(10, 5))
        return card

    def create_preview_card(self):
        card = self.create_card_frame("LIVE CROP PREVIEW")
        
        # Frame สำหรับรูป Preview โดยเฉพาะ (สีดำ)
        self.preview_box = ctk.CTkFrame(card, height=180, fg_color="black", corner_radius=5)
        self.preview_box.pack(fill="x", padx=10, pady=5)
        self.preview_box.pack_propagate(False)
        
        self.lbl_preview = ctk.CTkLabel(self.preview_box, text="WAITING...", text_color="gray")
        self.lbl_preview.place(relx=0.5, rely=0.5, anchor="center")

    def create_system_card(self):
        card = self.create_card_frame("SYSTEM CONTROL")
        
        # Model Select
        ctk.CTkLabel(card, text="AI Model:", font=FONT_LABEL).pack(anchor="w", padx=15)
        self.combo_model = ctk.CTkComboBox(card, values=[], command=self.load_selected_model, width=250)
        self.combo_model.pack(fill="x", padx=15, pady=(0, 10))
        
        # Buttons Row
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkButton(row, text="⚡ AUTO FOCUS", width=140, fg_color="#444", 
                      command=self.camera.trigger_autofocus).pack(side="left")
        ctk.CTkButton(row, text="💾 SAVE PRESET", width=140, fg_color="#444", 
                      command=self.save_current_preset).pack(side="right")
        
        # Load Preset
        ctk.CTkLabel(card, text="Load Preset:", font=FONT_LABEL).pack(anchor="w", padx=15, pady=(10,0))
        self.combo_preset = ctk.CTkComboBox(card, values=list(self.presets.keys()), command=self.apply_preset)
        self.combo_preset.pack(fill="x", padx=15, pady=5)

    def create_image_card(self):
        card = self.create_card_frame("IMAGE SETTINGS")
        
        self.sl_zoom = self.add_slider(card, "Zoom", 1.0, 4.0, 1.0)
        self.sl_bright = self.add_slider(card, "Brightness", -100, 100, 0)
        self.sl_cont = self.add_slider(card, "Contrast", 0.5, 3.0, 1.0)

    def create_recording_card(self):
        card = self.create_card_frame("DATASET CONFIG")
        
        # Class Name
        ctk.CTkLabel(card, text="Class Name (Output Folder):", font=FONT_LABEL).pack(anchor="w", padx=15)
        self.entry_class = ctk.CTkEntry(card, placeholder_text="e.g. pill_type_a")
        self.entry_class.pack(fill="x", padx=15, pady=5)
        
        # Path
        path_row = ctk.CTkFrame(card, fg_color="transparent")
        path_row.pack(fill="x", padx=10, pady=5)
        
        self.btn_path = ctk.CTkButton(path_row, text="📂 BROWSE", width=80, 
                                      fg_color="#555", command=self.change_folder)
        self.btn_path.pack(side="right")
        
        self.lbl_path = ctk.CTkLabel(path_row, text=".../Desktop", font=("Arial", 9), text_color="gray", anchor="w")
        self.lbl_path.pack(side="left", fill="x", expand=True)
        
        # Settings
        self.sl_interval = self.add_slider(card, "Save Every N Frames", 1, 60, 5)
        self.sl_conf = self.add_slider(card, "AI Confidence", 0.1, 1.0, 0.6)

    def add_slider(self, parent, label, min_v, max_v, default):
        # Header Row (Label + Value)
        head = ctk.CTkFrame(parent, fg_color="transparent")
        head.pack(fill="x", padx=15, pady=(5,0))
        
        ctk.CTkLabel(head, text=label, font=FONT_LABEL).pack(side="left")
        val_lbl = ctk.CTkLabel(head, text=str(default), font=FONT_VALUE, text_color=COLOR_ACCENT)
        val_lbl.pack(side="right")
        
        # Slider
        sl = ctk.CTkSlider(parent, from_=min_v, to=max_v, progress_color=COLOR_ACCENT)
        sl.set(default)
        sl.pack(fill="x", padx=10, pady=(0, 10))
        
        # Live Update Value
        def update_lbl(val):
            val_lbl.configure(text=f"{val:.1f}")
        sl.configure(command=update_lbl)
        
        return sl

    # ================= LOGIC LOOP =================
    def update_loop(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # 1. Processing
            zoom = self.sl_zoom.get()
            bright = self.sl_bright.get()
            cont = self.sl_cont.get()
            
            processed = self.processor.apply_filters(frame, zoom, bright, cont)
            display = processed.copy()
            self.processor.draw_crosshair(display)

            # 2. AI
            best_box = self.detector.predict(processed, conf=self.sl_conf.get())
            latest_crop = None

            status_text = "SYSTEM READY"
            
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cls_id = int(best_box.cls[0])
                conf = best_box.conf[0].item()
                
                # Safe Name
                name = self.detector.model.names.get(cls_id, f"ID_{cls_id}")
                
                # Draw
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{name} {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                status_text = f"DETECTED: {name} ({conf:.0%})"

                # Crop & Save
                crop = processed[y1:y2, x1:x2]
                if crop.size > 0:
                    latest_crop = self.processor.remove_background(crop)
                    if self.is_recording:
                        self.frame_count += 1
                        if self.frame_count % int(self.sl_interval.get()) == 0:
                            self.save_image(latest_crop)
                            status_text = f"SAVING... {self.count_saved}"

            # 3. UI Update
            self.show_video(display)
            self.show_preview(latest_crop)
            
            # Update Overlay Stats
            self.info_overlay.configure(text=f"{status_text}\nZOOM: {zoom:.1f}x\nFPS: N/A")

        self.after(20, self.update_loop)

    def show_video(self, img):
        # Resize to fit area
        h, w = img.shape[:2]
        vh = self.video_frame.winfo_height()
        vw = self.video_frame.winfo_width()
        
        if vh > 10 and vw > 10:
            scale = min(vh/h, vw/w)
            nh, nw = int(h*scale), int(w*scale)
            img = cv2.resize(img, (nw, nh))
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ctk_img = ctk.CTkImage(PIL.Image.fromarray(img_rgb), size=(nw, nh))
            self.lbl_video.configure(image=ctk_img, text="")
        
    def show_preview(self, img):
        if img is None: 
            self.lbl_preview.configure(image=None, text="WAITING...")
            return
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Fix aspect ratio for preview box (180px height)
        h, w = img.shape[:2]
        scale = 160 / h
        nw = int(w * scale)
        
        ctk_img = ctk.CTkImage(PIL.Image.fromarray(img_rgb), size=(nw, 160))
        self.lbl_preview.configure(image=ctk_img, text="")

    # ================= ACTIONS =================
    def save_image(self, img):
        cname = self.entry_class.get().strip() or "unknown"
        path = os.path.join(self.save_dir, cname)
        os.makedirs(path, exist_ok=True)
        fname = f"{int(time.time()*1000)}.png"
        cv2.imwrite(os.path.join(path, fname), img)
        self.count_saved += 1
        self.btn_record.configure(text=f"REC • {self.count_saved}")

    def toggle_record(self):
        if not self.is_recording:
            if not self.entry_class.get().strip():
                messagebox.showwarning("Warning", "Please enter Class Name!")
                return
            self.is_recording = True
            self.count_saved = 0
            self.btn_record.configure(text="STOP RECORDING", fg_color="#333", hover_color="#333")
            self.entry_class.configure(state="disabled")
        else:
            self.is_recording = False
            self.btn_record.configure(text="START RECORDING", fg_color=COLOR_RECORD, hover_color="#c0392b")
            self.entry_class.configure(state="normal")
            messagebox.showinfo("Done", f"Session saved {self.count_saved} images.")

    def refresh_models(self):
        models = self.detector.get_available_models()
        if models:
            self.combo_model.configure(values=models)
            self.combo_model.set(models[0])
            self.load_selected_model(models[0])
            
    def load_selected_model(self, name):
        success, msg = self.detector.load_model(os.path.join("models", name))
        print(msg)

    def change_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.save_dir = p
            self.lbl_path.configure(text=f".../{os.path.basename(p)}")

    def save_current_preset(self):
        name = simpledialog.askstring("Save", "Preset Name:")
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