from ui import DataCollectorApp
import customtkinter as ctk

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = DataCollectorApp()
    app.mainloop()