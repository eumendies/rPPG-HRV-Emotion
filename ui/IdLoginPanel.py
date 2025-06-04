from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox
from .color_const import MAIN_THEME, DARK_GRAY
from api import get_student_info


class IdLoginPanel(QWidget):
    switch_mode = pyqtSignal()
    login_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout()

        # 创建白色背景面板
        self.panel = QWidget()
        self.panel.setStyleSheet("background-color: white; border-radius: 20px;")
        self.panel_layout = QVBoxLayout()
        self.panel_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.panel_layout.setContentsMargins(30, 60, 30, 60)

        # 标题
        self.title_label = QLabel("非接触式情绪检测系统")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(f"font-size: 20px; color: {MAIN_THEME};")
        self.title_label.setFixedHeight(60)

        # 学号输入框
        self.id_label = QLabel("学号")
        self.id_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.id_label.setStyleSheet(f"color: {DARK_GRAY};")
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("请输入学号")
        self.id_input.setStyleSheet("padding: 10px; border-radius: 5px; border: 1px solid #ddd;color: #000000")
        self.id_input.setFixedHeight(40)

        # 用户名显示
        self.username_layout = QHBoxLayout()
        self.username_label = QLabel("张三")
        self.username_label.setStyleSheet(f"color: {MAIN_THEME};")
        self.username_label.setVisible(False)
        self.face_login_button = QPushButton("人脸登录")
        self.face_login_button.setStyleSheet(f"background-color: rgba(0,0,0,0); color: {MAIN_THEME};")
        self.face_login_button.clicked.connect(self.switch)
        self.username_layout.addWidget(self.username_label)
        self.username_layout.addStretch()
        self.username_layout.addWidget(self.face_login_button)

        # 分隔线
        self.separator = QLabel()
        self.separator.setStyleSheet("height: 1px;")

        # 登录按钮
        self.login_button = QPushButton("登录")
        self.login_button.setStyleSheet(
            f"background-color: {MAIN_THEME}; color: white; border: none; padding: 10px 20px; border-radius: 5px;")
        self.login_button.setFixedHeight(40)
        self.login_button.clicked.connect(self.login)

        # 添加到主布局
        self.panel_layout.addWidget(self.title_label)
        self.panel_layout.addWidget(self.id_label)
        self.panel_layout.addWidget(self.id_input)
        self.panel_layout.addLayout(self.username_layout)
        self.panel_layout.addWidget(self.separator)
        self.panel_layout.addWidget(self.login_button)
        self.panel.setLayout(self.panel_layout)

        self.main_layout.addWidget(self.panel)
        self.setLayout(self.main_layout)

    def switch(self):
        self.switch_mode.emit()

    def login(self):
        id = self.id_input.text()
        if not id.isdigit():
            QMessageBox.warning(self, "错误", "学号必须为纯数字")
            self.id_input.setText("")
            return

        _, info = get_student_info(int(id))
        if info is None or 'studentID' not in info:
            QMessageBox.warning(self, "错误", "学号不存在")
            self.id_input.setText("")
            return
        self.login_signal.emit(info['studentID'])
