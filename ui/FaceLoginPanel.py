from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QPushButton
from color_const import MAIN_THEME
from SquareWidget import SquareWidget
from SquareCamera import CameraWidget
from OverlapWidget import OverlayWidget
from assets import resource


class FaceLoginPanel(QWidget):
    switch_mode = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # 创建主布局
        main_layout = QHBoxLayout()

        # 创建白色背景面板
        panel = QWidget()
        panel.setStyleSheet("background-color: white; border-radius: 20px;")
        panel_layout = QVBoxLayout()
        panel_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.setContentsMargins(30, 60, 30, 60)

        # 标题
        title_label = QLabel("rPPG情绪检测系统")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"font-size: 20px; color: {MAIN_THEME};")
        title_label.setFixedHeight(60)
        panel_layout.addWidget(title_label)

        # 人脸识别区域
        square = SquareWidget(280, 5, MAIN_THEME)
        camera = CameraWidget(label_size=250, square_size=850)

        face_recognition_frame = QLabel()
        face_recognition_frame.setPixmap(QPixmap(":/imgs/detection.png"))
        face_recognition_frame.setScaledContents(True)
        face_recognition_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        face_recognition_frame.setFixedSize(250, 200)
        face_recognition_frame.setStyleSheet(f"background-color: rgba(0,0,0,0);")

        camera_and_frame = OverlayWidget(camera, face_recognition_frame)

        detection_zone = OverlayWidget(square, camera_and_frame)

        panel_layout.addWidget(detection_zone)

        # 用户信息
        info_layout = QHBoxLayout()
        info_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        username_label = QLabel("张三")
        username_label.setObjectName("张三")
        username_label.setStyleSheet(f"color: {MAIN_THEME};")

        student_id_label = QLabel("202501014")
        student_id_label.setStyleSheet(f"color: {MAIN_THEME};")

        student_login_button = QPushButton("学号登录")
        student_login_button.setStyleSheet(f"background-color: rgba(0, 0, 0, 0); color: {MAIN_THEME};")
        student_login_button.clicked.connect(self.switch)

        info_layout.addWidget(username_label)
        info_layout.addWidget(student_id_label)
        info_layout.addWidget(student_login_button)
        panel_layout.addLayout(info_layout)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #eee;")
        panel_layout.addWidget(separator)

        # 登录按钮
        login_button = QPushButton("登录")
        login_button.setStyleSheet(
            "background-color: #4a86e8; color: white; border: none; padding: 10px 20px; border-radius: 5px;")
        login_button.setFixedHeight(40)
        panel_layout.addWidget(login_button)

        panel.setLayout(panel_layout)
        main_layout.addWidget(panel)
        self.setLayout(main_layout)

    def switch(self):
        self.switch_mode.emit()