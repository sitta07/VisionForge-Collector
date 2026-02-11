import tkinter as tk
from ui import DataCollectorApp

if __name__ == "__main__":
    # Setup High DPI for Windows (Optional but looks good)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    app = DataCollectorApp(root, "SnapSet-Pro: Modular Edition")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()