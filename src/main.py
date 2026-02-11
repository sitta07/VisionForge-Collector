import sys
from PySide6.QtWidgets import QApplication
from ui_qt import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Optional: Fix High DPI scaling
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())