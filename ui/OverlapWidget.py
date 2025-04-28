from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import (QWidget, QVBoxLayout)


class OverlayWidget(QWidget):
    def __init__(self, base_widget, overlay_widget, parent=None):
        super().__init__(parent)
        self.base_widget = base_widget  # 底层控件（CornerSquare）
        self.overlay_widget = overlay_widget  # 叠加控件

        # 使用垂直布局管理基础控件
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.base_widget)

        # 将叠加控件添加到同一容器
        self.overlay_widget.setParent(self)
        self.overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents)  # 穿透鼠标事件

    def resizeEvent(self, event):
        """自动居中叠加控件"""
        base_rect = self.base_widget.geometry()

        # 计算居中位置
        x = base_rect.x() + (base_rect.width() - self.overlay_widget.width()) // 2
        y = base_rect.y() + (base_rect.height() - self.overlay_widget.height()) // 2

        # 设置叠加控件位置
        self.overlay_widget.setGeometry(QRect(x, y,
                                              self.overlay_widget.width(),
                                              self.overlay_widget.height()))
        super().resizeEvent(event)

    def width(self):
        return max(self.base_widget.width(), self.overlay_widget.width())

    def height(self):
        return max(self.base_widget.height(), self.overlay_widget.height())
