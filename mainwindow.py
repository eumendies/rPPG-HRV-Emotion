import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget


class Ui_MainWindow(object):
    def setup_ui(self, MainWindow):
        MainWindow.setObjectName("rPPG-HRV情绪识别系统")
        MainWindow.resize(720, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.main_layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()
        self.operations_info_layout = QVBoxLayout()
        self.buttons_layout = QHBoxLayout()

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

        # 多选框，用于选择采集数据量
        self.comboBox_data_num = QComboBox()
        self.comboBox_data_num.setMinimumSize(QtCore.QSize(0, 28))
        self.comboBox_data_num.setMaximumSize(QtCore.QSize(16777215, 28))
        self.comboBox_data_num.addItem("采集数据量(默认256帧)")
        self.comboBox_data_num.addItem("256")
        self.comboBox_data_num.addItem("512")
        self.comboBox_data_num.addItem("1024")
        self.comboBox_data_num.addItem("2048")

        # 多选框，用于选择心率计算方式
        self.comboBox_hr_method = QComboBox()
        self.comboBox_hr_method.setMinimumSize(QtCore.QSize(0, 28))
        self.comboBox_hr_method.setMaximumSize(QtCore.QSize(16777215, 28))
        self.comboBox_hr_method.addItem("心率计算方式(默认FFT)")
        self.comboBox_hr_method.addItem("FFT")
        self.comboBox_hr_method.addItem("Peak")

        # 显示原始信号、滤波信号的按钮
        self.Button_Raw = QPushButton()
        self.Button_Raw.setText("原始信号")
        self.Button_Filtered = QPushButton()
        self.Button_Filtered.setText("滤波信号")

        self.buttons_layout.addWidget(self.comboBox)
        self.buttons_layout.addWidget(self.comboBox_data_num)
        self.buttons_layout.addWidget(self.comboBox_hr_method)
        self.buttons_layout.addWidget(self.Button_Raw)
        self.buttons_layout.addWidget(self.Button_Filtered)

        # 显示FPS、BPM等信息的标签
        self.info_label = QLabel()
        self.info_label.setMinimumSize(QtCore.QSize(0, 50))
        self.info_label.setFont(QFont("Consolas", 16))
        self.info_label.setText("")
        self.info_label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.info_label.setObjectName("info")

        self.info_table = QTableWidget(2, 4)
        self.info_table.setHorizontalHeaderLabels(["前额测量心率", "左脸颊测量心率", "右脸颊测量心率", "最终心率"])
        self.info_table.setVerticalHeaderLabels(["数值", "置信度"])
        self.info_table.setMinimumHeight(80)

        # 时域HRV
        self.time_hrv_table = QTableWidget(3, 5)
        self.time_hrv_table.setHorizontalHeaderLabels(["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_pNN50"])
        self.time_hrv_table.setVerticalHeaderLabels(["前额", "左脸颊", "右脸颊"])
        self.time_hrv_table.setMinimumHeight(120)

        # 频域HRV
        self.freq_hrv_table = QTableWidget(3, 6)
        self.freq_hrv_table.setHorizontalHeaderLabels(["HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF", "HRV_TP"])
        self.freq_hrv_table.setVerticalHeaderLabels(["前额", "左脸颊", "右脸颊"])
        self.freq_hrv_table.setMinimumWidth(650)
        self.freq_hrv_table.setMinimumHeight(120)

        self.operations_info_layout.addLayout(self.buttons_layout)
        self.operations_info_layout.addWidget(self.info_table)
        self.operations_info_layout.addWidget(self.time_hrv_table)
        self.operations_info_layout.addWidget(self.freq_hrv_table)
        self.operations_info_layout.addWidget(self.info_label)

        # 摄像头画面
        self.face = QtWidgets.QLabel(self.centralwidget)
        self.face.setFixedSize(320, 180)
        self.face.setScaledContents(True)

        self.top_layout.addSpacing(10)
        self.top_layout.addLayout(self.operations_info_layout)
        self.top_layout.addSpacing(10)
        self.top_layout.addWidget(self.face)
        self.top_layout.addSpacing(10)

        self.graphs_layout = QHBoxLayout()
        # 用于展示各种信号
        self.Layout_BVP = QVBoxLayout()
        self.Layout_BVP.setContentsMargins(0, 0, 0, 0)
        self.Layout_Signal = QVBoxLayout()
        self.Layout_Signal.setContentsMargins(0, 0, 0, 0)
        self.Layout_Spec = QVBoxLayout()
        self.Layout_Spec.setContentsMargins(0, 0, 0, 0)

        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)

        # bvp信号图
        self.Signal_fore = pg.PlotWidget(MainWindow)
        self.Signal_left = pg.PlotWidget(MainWindow)
        self.Signal_right = pg.PlotWidget(MainWindow)

        self.Sig_f = self.Signal_fore.plot()
        self.Sig_l = self.Signal_left.plot()
        self.Sig_r = self.Signal_right.plot()

        self.label_sig_fore = QLabel("额头信号")
        self.label_sig_left = QLabel("左脸颊信号")
        self.label_sig_right = QLabel("右脸颊信号")
        self.label_sig_fore.setFont(font)
        self.label_sig_left.setFont(font)
        self.label_sig_right.setFont(font)

        self.Layout_BVP.addWidget(self.label_sig_fore)
        self.Layout_BVP.addWidget(self.Signal_fore)
        self.Layout_BVP.addWidget(self.label_sig_left)
        self.Layout_BVP.addWidget(self.Signal_left)
        self.Layout_BVP.addWidget(self.label_sig_right)
        self.Layout_BVP.addWidget(self.Signal_right)

        # 信号频域图
        self.Spectrum_fore = pg.PlotWidget(self)
        self.Spectrum_left = pg.PlotWidget(self)
        self.Spectrum_right = pg.PlotWidget(self)

        self.Spec_f = self.Spectrum_fore.plot()
        self.Spec_l = self.Spectrum_left.plot()
        self.Spec_r = self.Spectrum_right.plot()

        self.label_spec_fore = QLabel("前额信号频域图")
        self.label_spec_left = QLabel("左脸颊信号频域图")
        self.label_spec_right = QLabel("右脸颊信号频域图")
        self.label_spec_fore.setFont(font)
        self.label_spec_left.setFont(font)
        self.label_spec_right.setFont(font)

        self.Layout_Spec.addWidget(self.label_spec_fore)
        self.Layout_Spec.addWidget(self.Spectrum_fore)
        self.Layout_Spec.addWidget(self.label_spec_left)
        self.Layout_Spec.addWidget(self.Spectrum_left)
        self.Layout_Spec.addWidget(self.label_spec_right)
        self.Layout_Spec.addWidget(self.Spectrum_right)

        # RGB直方图
        self.Hist_fore = pg.PlotWidget(self)
        self.Hist_left = pg.PlotWidget(self)
        self.Hist_right = pg.PlotWidget(self)

        self.Hist_fore.setYRange(0, 0.2)
        self.Hist_left.setYRange(0, 0.2)
        self.Hist_right.setYRange(0, 0.2)

        self.Hist_fore_r = self.Hist_fore.plot()
        self.Hist_fore_g = self.Hist_fore.plot()
        self.Hist_fore_b = self.Hist_fore.plot()
        self.Hist_left_r = self.Hist_left.plot()
        self.Hist_left_g = self.Hist_left.plot()
        self.Hist_left_b = self.Hist_left.plot()
        self.Hist_right_r = self.Hist_right.plot()
        self.Hist_right_g = self.Hist_right.plot()
        self.Hist_right_b = self.Hist_right.plot()

        self.label_hist_fore = QLabel("前额RGB直方图")
        self.label_hist_left = QLabel("左脸颊RGB直方图")
        self.label_hist_right = QLabel("右脸颊RGB直方图")
        self.label_hist_fore.setFont(font)
        self.label_hist_left.setFont(font)
        self.label_hist_right.setFont(font)

        self.Layout_Signal.addWidget(self.label_hist_fore)
        self.Layout_Signal.addWidget(self.Hist_fore)
        self.Layout_Signal.addWidget(self.label_hist_left)
        self.Layout_Signal.addWidget(self.Hist_left)
        self.Layout_Signal.addWidget(self.label_hist_right)
        self.Layout_Signal.addWidget(self.Hist_right)

        self.graphs_layout.addLayout(self.Layout_BVP)
        self.graphs_layout.addLayout(self.Layout_Spec)
        self.graphs_layout.addLayout(self.Layout_Signal)

        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.graphs_layout)
        self.main_layout.setSpacing(5)

        MainWindow.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(self.main_layout)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
