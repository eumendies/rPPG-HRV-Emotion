import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor


class SquareWidget(QWidget):
    def __init__(self, side_length=200, thickness=1, color="#2196F3", parent=None):
        super().__init__(parent)
        self.square_size = side_length  # 正方形边长
        self.line_thickness = thickness  # 边线粗细
        self.line_color = QColor(color)  # 边线颜色
        self.setFixedSize(side_length, side_length)  # 固定控件大小

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 设置画笔属性
        pen = QPen()
        pen.setColor(self.line_color)
        pen.setWidth(self.line_thickness)
        pen.setCapStyle(Qt.SquareCap)
        painter.setPen(pen)

        # 计算线段参数
        L = self.square_size
        T = self.line_thickness
        segment = L // 4  # 保留1/4长度的边

        # 绘制四个角
        # 左上角（水平右 + 垂直下）
        painter.drawLine(0, 0, segment, 0)
        painter.drawLine(0, 0, 0, segment)

        # 右上角（水平左 + 垂直下）
        painter.drawLine(L, 0, L - segment, 0)
        painter.drawLine(L, 0, L, segment)

        # 左下角（水平右 + 垂直上）
        painter.drawLine(0, L, segment, L)
        painter.drawLine(0, L, 0, L - segment)

        # 右下角（水平左 + 垂直上）
        painter.drawLine(L, L, L - segment, L)
        painter.drawLine(L, L, L, L - segment)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Corner Square Demo')
        self.setGeometry(300, 300, 400, 400)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 创建包含正方形的水平布局（用于演示居中）
        h_layout = QHBoxLayout()
        h_layout.addStretch()

        # 实例化自定义控件（可调整参数）
        square = SquareWidget(
            side_length=200,
            thickness=1,
            color="#FF5722"
        )

        h_layout.addWidget(square)
        h_layout.addStretch()

        main_layout.addLayout(h_layout)
        self.setLayout(main_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())