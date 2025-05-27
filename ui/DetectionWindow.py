import os
import sys
import threading
import uuid
from queue import Queue

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, \
    QStackedWidget, QMessageBox

from .Background import LowPolyBackground
from .OverlapWidget import OverlayWidget
from .Progress import CircularProgress
from .SquareWidget import SquareWidget
from .color_const import MAIN_THEME
from .assets import resource
from api import send_process_data, upload_video_file


class DetectionWindow(QMainWindow):
    def __init__(self, series_class=None, crop_size=850):
        super().__init__()
        self.student_id = None
        self.detection_id = None
        self.setWindowTitle("检测")
        self.setGeometry(100, 100, 1080, 720)
        self.init_ui()

        self.crop_size = crop_size
        self.series_class = series_class
        self.series_class.image_signal.connect(self.display_image)
        self.series_class.undetected_signal.connect(self.clear_frame)

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
        self.clear_frame()
        self.series_class.start()
        self.series_class.change_data_num(256)
        self.stack_button_prompt.setCurrentIndex(1)
        self.detection_id = uuid.uuid4().hex

    def pause_detection(self):
        self.series_class.stop()
        self.face.clear()
        self.progress_bar.update_progress(0)
        thread = threading.Thread(target=self.output_video)
        thread.start()

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
            signals = np.around(signals, decimals=4)
            send_process_data(self.detection_id, self.student_id, signals.tolist())

            self.pause_detection()
            self.stack_button_prompt.setCurrentIndex(0)
            # 创建消息框
            msg_box = QMessageBox()
            msg_box.setText("检测完毕")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.button(QMessageBox.Ok).setText("确定")
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    font-size: 16px;
                    color: #4a86e8;
                }
                QPushButton {
                    background-color: #4a86e8;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #5a96f8;
                }
            """)
            msg_box.exec_()

    def output_video(self):
        video_name = f"student_{self.student_id}.mp4"
        video_path = os.path.join("output", video_name)
        video_writer = cv2.VideoWriter(video_path, self.fourcc, 30, (self.crop_size, self.crop_size))

        first_frame_saved = False  # 用于判断是否已保存首帧
        first_frame_path = None

        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
            if not first_frame_saved:
                first_frame_path = os.path.join("output", f"student_{self.student_id}_first_frame.png")
                cv2.imwrite(first_frame_path, frame)  # 保存首帧
                first_frame_saved = True
            video_writer.write(frame)
        self.video_count += 1
        video_writer.release()
        result = upload_video_file(self.student_id, self.detection_id, video_path, first_frame_path)
        if "error" not in result:
            os.remove(video_path)
            os.remove(first_frame_path)
        else:
            print("上传失败")
        
    def set_student_id(self, student_id):
        self.student_id = student_id

    def clear_frame(self):
        self.frame_queue.queue.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())