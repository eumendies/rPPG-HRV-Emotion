from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget
from PyQt5.QtGui import QFont


class Ui_MainWindow(object):
    def setup_ui(self, MainWindow):
        MainWindow.setObjectName("rPPG-HRV情绪识别系统")
        MainWindow.resize(720, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.main_layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()
        self.operations_info_layout = QVBoxLayout()

        # 多选框，用于选择信号计算方式
        self.comboBox = QComboBox()
        self.comboBox.setMinimumSize(QtCore.QSize(0, 28))
        self.comboBox.setMaximumSize(QtCore.QSize(16777215, 28))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("GREEN")
        self.comboBox.addItem("GREEN-RED")
        self.comboBox.addItem("CHROM")
        self.comboBox.addItem("PBV")
        self.comboBox.addItem("POS")
        self.comboBox.setMinimumWidth(350)

        # 显示原始信号、滤波信号的按钮
        self.Button_Raw = QPushButton()
        self.Button_Raw.setText("原始信号")
        self.Button_Filtered = QPushButton()
        self.Button_Filtered.setText("滤波信号")

        # 显示FPS、BPM等信息的标签
        self.info_label = QLabel()
        self.info_label.setMinimumSize(QtCore.QSize(0, 220))
        self.info_label.setFont(QFont("Consolas", 16))
        self.info_label.setText("")
        self.info_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.info_label.setObjectName("info")

        self.operations_info_layout.addWidget(self.comboBox)
        self.operations_info_layout.addWidget(self.Button_Raw)
        self.operations_info_layout.addWidget(self.Button_Filtered)
        self.operations_info_layout.addWidget(self.info_label)

        # 摄像头画面
        self.face = QtWidgets.QLabel(self.centralwidget)
        self.face.setFixedSize(320, 180)
        self.face.setScaledContents(True)

        self.top_layout.addLayout(self.operations_info_layout)
        self.top_layout.addStretch()
        self.top_layout.addWidget(self.face)

        self.graphs_layout = QHBoxLayout()
        # 用于展示各种信号
        self.Layout_BVP = QVBoxLayout()
        self.Layout_BVP.setContentsMargins(0, 0, 0, 0)
        self.Layout_Signal = QVBoxLayout()
        self.Layout_Signal.setContentsMargins(0, 0, 0, 0)
        self.Layout_Spec = QVBoxLayout()
        self.Layout_Spec.setContentsMargins(0, 0, 0, 0)

        self.graphs_layout.addLayout(self.Layout_BVP)
        self.graphs_layout.addLayout(self.Layout_Spec)
        self.graphs_layout.addLayout(self.Layout_Signal)

        # 时域HRV
        self.time_hrv_table = QTableWidget(3, 5)
        self.time_hrv_table.setHorizontalHeaderLabels(["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_pNN50"])
        self.time_hrv_table.setVerticalHeaderLabels(["前额", "左脸颊", "右脸颊"])

        # 频域HRV
        self.freq_hrv_table = QTableWidget(3, 6)
        self.freq_hrv_table.setHorizontalHeaderLabels(["HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF", "HRV_TP"])
        self.freq_hrv_table.setVerticalHeaderLabels(["前额", "左脸颊", "右脸颊"])

        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.graphs_layout)
        self.main_layout.addWidget(self.time_hrv_table)
        self.main_layout.addWidget(self.freq_hrv_table)
        self.main_layout.setSpacing(5)

        MainWindow.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(self.main_layout)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
