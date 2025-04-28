import sys

from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication, QWidget
from color_const import MAIN_THEME


class CircularProgress(QWidget):
    def __init__(self, progress_width=5):
        super().__init__()

        # 进度条参数
        self.progress = 0
        self.max_progress = 100
        self.progress_width = progress_width
        self.text_color = QColor(74, 134, 232)

        # 设置定时器更新进度
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)

    def update_progress(self):
        self.progress += 0.03
        if self.progress > self.max_progress:
            self.progress = 0
        self.update()

    def paintEvent(self, event):
        width = self.width()
        height = self.height()
        length = min(width, height)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(width / 2, height / 2)

        # 绘制背景圆环
        painter.save()
        pen = QPen(QColor(230, 230, 230), self.progress_width)
        painter.setPen(pen)
        painter.drawEllipse(QRectF(-length / 2.5, -length / 2.5, length * 2 / 2.5, length * 2 / 2.5))
        painter.restore()

        # 绘制进度圆环
        painter.save()
        pen = QPen(QColor(MAIN_THEME), self.progress_width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(QRectF(-length / 2.5, -length / 2.5, length * 2 / 2.5, length * 2 / 2.5), 90 * 16, -int(360 * self.progress / self.max_progress) * 16)
        painter.restore()

        # 绘制进度文本
        # painter.save()
        # painter.setFont(self.font())
        # painter.setFont(QFont('Arial', self.font_size))
        # painter.setPen(self.text_color)
        # painter.drawText(QRectF(-width / 2, -height / 2, width, height), Qt.AlignCenter, f"{self.progress}%")
        # painter.restore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CircularProgress()
    window.setFixedSize(100, 100)
    window.show()
    sys.exit(app.exec_())