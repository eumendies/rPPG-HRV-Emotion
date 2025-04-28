import sys

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication


class CameraWidget(QWidget):
    def __init__(self, label_size, square_size):
        super().__init__()
        self.label_size = label_size
        self.square_size = square_size
        self.init_ui()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        self.label = QLabel(self)
        self.label.setFixedSize(self.label_size, self.label_size)
        self.label.setScaledContents(True)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setContentsMargins(0, 0, 0, 0)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.crop_to_square(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.label.setPixmap(pixmap)

    def crop_to_square(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        half_size = self.square_size // 2
        x1, y1 = center_x - half_size, center_y - half_size
        x2, y2 = center_x + half_size, center_y + half_size
        return frame[y1:y2, x1:x2]

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()

    def width(self):
        return self.label_size

    def height(self):
        return self.label_size


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = CameraWidget(label_size=200, square_size=850)  # Set the size of the square region
    widget.show()
    sys.exit(app.exec_())