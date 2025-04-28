import random

import numpy as np
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPainter, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget
from scipy.spatial import Delaunay

from color_const import MAIN_THEME, MAIN_THEME_DARK


class LowPolyBackground(QWidget):
    def __init__(self, point_count=60):
        super().__init__()
        self.point_count = point_count
        self._generate_mesh()

    def _generate_mesh(self):
        w, h = self.width(), self.height()
        # 1) 随机点 + 四个角点
        pts = np.random.rand(self.point_count, 2) * [w, h]
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
        points = np.vstack([pts, corners])

        # 2) Delaunay 三角剖分
        tri = Delaunay(points)

        # 3) 配色：深浅蓝两色
        color_dark = QColor(MAIN_THEME_DARK)  # 深蓝
        color_light = QColor(MAIN_THEME)  # 浅蓝

        self.triangles = []
        for simplex in tri.simplices:
            coords = points[simplex]
            polygon = [QPointF(x, y) for x, y in coords]

            t = random.random()
            r = color_dark.red() + t * (color_light.red() - color_dark.red())
            g = color_dark.green() + t * (color_light.green() - color_dark.green())
            b = color_dark.blue() + t * (color_light.blue() - color_dark.blue())
            color = QColor(int(r), int(g), int(b))

            self.triangles.append((QPolygonF(polygon), color))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for poly, color in self.triangles:
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(poly)

    def resizeEvent(self, event):
        # 窗口大小改变时，重新生成适配新尺寸的多边形
        self._generate_mesh()
        self.update()
