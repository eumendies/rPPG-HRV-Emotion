import sys

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QApplication, QStackedWidget, QMainWindow

from .Background import LowPolyBackground
from .FaceLoginPanel import FaceLoginPanel
from .IdLoginPanel import IdLoginPanel
from .DetectionWindow import DetectionWindow


class LoginWindow(QMainWindow):
    close_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录")
        self.setGeometry(100, 100, 1080, 720)

        self.bg = LowPolyBackground(point_count=80)
        self.setCentralWidget(self.bg)

        # 创建主布局
        self.main_layout = QHBoxLayout(self.bg)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.login_panel = QStackedWidget()
        self.login_panel.setFixedHeight(600)
        self.id_login_panel = IdLoginPanel()
        self.face_login_panel = FaceLoginPanel()
        self.id_login_panel.switch_mode.connect(self.show_face_login)
        self.face_login_panel.switch_mode.connect(self.show_id_login)
        self.login_panel.addWidget(self.id_login_panel)
        self.login_panel.addWidget(self.face_login_panel)

        self.id_login_panel.login_signal.connect(self.login)
        self.face_login_panel.login_signal.connect(self.login)

        self.main_layout.addStretch()
        self.main_layout.addWidget(self.login_panel)
        self.main_layout.addStretch()

    def login(self):
        self.close_signal.emit()
        self.close()

    def show_face_login(self):
        self.login_panel.setCurrentIndex(1)
        self.face_login_panel.start_camera()

    def show_id_login(self):
        self.login_panel.setCurrentIndex(0)
        self.face_login_panel.close_camera()

    def closeEvent(self, a0):
        self.face_login_panel.close_camera()
        super().closeEvent(a0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    new_window = DetectionWindow()
    window.close_signal.connect(new_window.show)
    sys.exit(app.exec())
