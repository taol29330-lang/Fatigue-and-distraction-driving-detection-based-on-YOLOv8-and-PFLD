# src/main.py
import sys
from PySide6.QtWidgets import QApplication
from app import DriverMonitorWindow

def main():
    app = QApplication(sys.argv)
    w = DriverMonitorWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()