from ui import DetectionWindow, LoginWindow
import sys
from PyQt5.QtWidgets import QApplication
from face2series import CAM2FACE


if __name__ == '__main__':
    # 画面转rgb信号
    series_class = CAM2FACE()
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()

    new_window = DetectionWindow(series_class=series_class)
    window.close_signal.connect(new_window.show)
    sys.exit(app.exec())
