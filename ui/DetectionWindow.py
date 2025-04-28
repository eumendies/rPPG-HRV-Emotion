import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QProgressBar, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout

from Background import LowPolyBackground
from OverlapWidget import OverlayWidget
from SquareCamera import CameraWidget
from SquareWidget import SquareWidget
from color_const import MAIN_THEME
from assets import resource
from Progress import CircularProgress


class DetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("检测")
        self.setGeometry(100, 100, 1080, 720)

        self.bg = LowPolyBackground(point_count=80)
        self.setCentralWidget(self.bg)

        # 创建布局
        layout = QHBoxLayout(self.bg)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 创建白色背景面板
        panel = QWidget()
        panel.setStyleSheet("background-color: white; border-radius: 20px;")
        panel_layout = QVBoxLayout()
        panel_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.setContentsMargins(30, 60, 30, 60)

        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setValue(45)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #000;
                border-radius: 10px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #000;
                border-radius: 10px;
            }
        """)

        # 人脸识别区域
        square = SquareWidget(280, 5, MAIN_THEME)
        progress = CircularProgress(progress_width=15)
        progress.setFixedSize(550, 550)
        camera = CameraWidget(label_size=250, square_size=850)

        # 辅助框
        face_recognition_frame = QLabel()
        face_recognition_frame.setPixmap(QPixmap(":/imgs/detection.png"))
        face_recognition_frame.setScaledContents(True)
        face_recognition_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        face_recognition_frame.setFixedSize(250, 200)
        face_recognition_frame.setStyleSheet(f"background-color: rgba(0,0,0,0);")

        camera_and_frame = OverlayWidget(camera, face_recognition_frame)
        detection_zone = OverlayWidget(square, camera_and_frame)
        detection_zone = OverlayWidget(progress, detection_zone)
        panel_layout.addWidget(detection_zone, 0, Qt.AlignmentFlag.AlignCenter)

        # 设置提示文本
        prompt_text = QLabel("请在1分钟固定在像框内保持稳定")
        prompt_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt_text.setStyleSheet(f"font-size: 18px; color: {MAIN_THEME};")

        panel_layout.addWidget(prompt_text)
        panel.setLayout(panel_layout)

        layout.addStretch()
        layout.addWidget(panel)
        layout.addStretch()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())