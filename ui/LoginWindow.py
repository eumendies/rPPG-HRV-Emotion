import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QApplication, QStackedWidget, QMainWindow

from IdLoginPanel import IdLoginPanel
from FaceLoginPanel import FaceLoginPanel
from color_const import MAIN_THEME_DARK
from Background import LowPolyBackground


class LoginWindow(QMainWindow):
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

        self.main_layout.addStretch()
        self.main_layout.addWidget(self.login_panel)
        self.main_layout.addStretch()

        self.setLayout(self.main_layout)

    def login(self):
        self.username_label.setVisible(True)

    def show_face_login(self):
        self.login_panel.setCurrentIndex(1)

    def show_id_login(self):
        self.login_panel.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec())
