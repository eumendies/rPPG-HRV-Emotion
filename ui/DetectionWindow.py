import os
import sys
from queue import Queue

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, \
    QStackedWidget

from .Background import LowPolyBackground
from .OverlapWidget import OverlayWidget
from .Progress import CircularProgress
from .SquareWidget import SquareWidget
from .color_const import MAIN_THEME
from .assets import resource


class DetectionWindow(QMainWindow):
    def __init__(self, series_class=None, crop_size=850):
        super().__init__()
        self.setWindowTitle("检测")
        self.setGeometry(100, 100, 1080, 720)
        self.init_ui()

        self.crop_size = crop_size
        self.series_class = series_class
        self.series_class.image_signal.connect(self.display_image)

        # 帧队列，检测完毕后保存视频
        self.frame_queue = Queue()
        self.video_count = 0
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def init_ui(self):
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

        # 人脸识别区域
        square = SquareWidget(280, 5, MAIN_THEME)
        self.progress_bar = CircularProgress(progress_width=15)
        self.progress_bar.setFixedSize(550, 550)

        self.face = QLabel()
        self.face.setFixedSize(250, 250)
        self.face.setScaledContents(True)

        # 辅助框
        face_recognition_frame = QLabel()
        face_recognition_frame.setPixmap(QPixmap(":/imgs/detection.png"))
        face_recognition_frame.setScaledContents(True)
        face_recognition_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        face_recognition_frame.setFixedSize(250, 200)
        face_recognition_frame.setStyleSheet(f"background-color: rgba(0,0,0,0);")

        camera_and_frame = OverlayWidget(self.face, face_recognition_frame)
        detection_zone = OverlayWidget(square, camera_and_frame)
        detection_zone = OverlayWidget(self.progress_bar, detection_zone)
        panel_layout.addWidget(detection_zone, 0, Qt.AlignmentFlag.AlignCenter)

        start_button = QPushButton("开始检测")
        start_button.setStyleSheet(
            f"background-color: {MAIN_THEME}; color: white; border: none; padding: 10px 20px; border-radius: 5px;")
        start_button.clicked.connect(self.start_detection)

        # 设置提示文本
        prompt_text = QLabel("请在1分钟固定在像框内保持稳定")
        prompt_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt_text.setStyleSheet(f"font-size: 18px; color: {MAIN_THEME};")

        self.stack_button_prompt = QStackedWidget()
        self.stack_button_prompt.addWidget(start_button)
        self.stack_button_prompt.addWidget(prompt_text)

        panel_layout.addWidget(self.stack_button_prompt)
        panel.setLayout(panel_layout)

        layout.addStretch()
        layout.addWidget(panel)
        layout.addStretch()

    def start_detection(self):
        self.series_class.start()
        self.series_class.change_data_num(1024)
        self.stack_button_prompt.setCurrentIndex(1)

    def pause_detection(self):
        self.series_class.stop()
        self.face.setPixmap(QPixmap())
        self.progress_bar.update_progress(0)
        self.output_video()

    def crop_to_square(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        half_size = self.crop_size // 2
        x1, y1 = center_x - half_size, center_y - half_size
        x2, y2 = center_x + half_size, center_y + half_size
        return frame[y1:y2, x1:x2, :]

    def display_image(self, numbered_frame):
        if numbered_frame is not None:
            frame = numbered_frame.frame
            frame = cv2.flip(frame, 1)
            frame = self.crop_to_square(frame)
            self.frame_queue.put(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.face.setPixmap(QPixmap.fromImage(qimg))
            self.update_progress()

    def update_progress(self):
        progress = self.series_class.get_progress() * 100
        self.progress_bar.update_progress(progress)
        if progress >= 100:
            signals = self.series_class.get_signals()
            self.pause_detection()
            self.stack_button_prompt.setCurrentIndex(0)

    def output_video(self):
        video_name = f"student_{self.video_count}.mp4"
        video_path = os.path.join("output", video_name)
        video_writer = cv2.VideoWriter(video_path, self.fourcc, 30, (self.crop_size, self.crop_size))
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
            video_writer.write(frame)
        self.video_count += 1
        video_writer.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())