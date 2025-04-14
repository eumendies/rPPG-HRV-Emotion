"""
Author: Harryhht
Date: 2022-02-01 20:51:25
LastEditors: Eumendies
LastEditTime: 2025-04-10 14:30:50
Description: Main structure for the application
"""

import sys
import time

from mainwindow import Ui_MainWindow

from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pyqtgraph as pg

from obspy.signal.detrend import spline
from scipy import signal
import numpy as np
import cv2 as cv
from series2rPPG import Series2rPPG
from face2series import CAM2FACE
from constants import RED_PEN, GREEN_PEN, BLUE_PEN, ONE_MINUTE

MIN_HZ = 0.83  # 50 BPM - minimum allowed heart rate
MAX_HZ = 2.5  # 150 BPM - maximum allowed heart rate


class MainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWin, self).__init__(parent)
        self.setupUi(self)

        self.Hist_fore = pg.PlotWidget(self)
        self.Hist_left = pg.PlotWidget(self)
        self.Hist_right = pg.PlotWidget(self)

        self.Hist_fore.setYRange(0, 0.2)
        self.Hist_left.setYRange(0, 0.2)
        self.Hist_right.setYRange(0, 0.2)

        self.label_fore = QLabel(self.verticalLayoutWidget)
        self.label_left = QLabel(self.verticalLayoutWidget)
        self.label_right = QLabel(self.verticalLayoutWidget)
        self.Hist_fore_r = self.Hist_fore.plot()
        self.Hist_fore_g = self.Hist_fore.plot()
        self.Hist_fore_b = self.Hist_fore.plot()
        self.Hist_left_r = self.Hist_left.plot()
        self.Hist_left_g = self.Hist_left.plot()
        self.Hist_left_b = self.Hist_left.plot()
        self.Hist_right_r = self.Hist_right.plot()
        self.Hist_right_g = self.Hist_right.plot()
        self.Hist_right_b = self.Hist_right.plot()
        self.Layout_Signal.addWidget(self.Hist_fore)
        self.Layout_Signal.addWidget(self.Hist_left)
        self.Layout_Signal.addWidget(self.Hist_right)

        self.Signal_fore = pg.PlotWidget(self)
        self.Signal_left = pg.PlotWidget(self)
        self.Signal_right = pg.PlotWidget(self)

        self.Sig_f = self.Signal_fore.plot()
        self.Sig_l = self.Signal_left.plot()
        self.Sig_r = self.Signal_right.plot()

        self.Spectrum_fore = pg.PlotWidget(self)
        self.Spectrum_left = pg.PlotWidget(self)
        self.Spectrum_right = pg.PlotWidget(self)

        self.Spec_f = self.Spectrum_fore.plot()
        self.Spec_l = self.Spectrum_left.plot()
        self.Spec_r = self.Spectrum_right.plot()

        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)

        self.label_fore.setFont(font)
        self.label_fore.setText("额头信号")
        self.Layout_BVP.addWidget(self.label_fore)

        self.Layout_BVP.addWidget(self.Signal_fore)

        self.label_left.setFont(font)
        self.label_left.setText("左脸颊信号")
        self.Layout_BVP.addWidget(self.label_left)

        self.Layout_BVP.addWidget(self.Signal_left)

        self.label_right.setFont(font)
        self.label_right.setText("右脸颊信号")
        self.Layout_BVP.addWidget(self.label_right)

        self.Layout_BVP.addWidget(self.Signal_right)

        self.Layout_Spec.addWidget(self.Spectrum_fore)
        self.Layout_Spec.addWidget(self.Spectrum_left)
        self.Layout_Spec.addWidget(self.Spectrum_right)

        self.face.setScaledContents(True)
        self.processor = Series2rPPG()
        self.series_class = CAM2FACE()
        self.series_class.PROCESS_start()

        # 用于更新画面的定时器
        self.TIMER_Frame = QTimer()
        self.TIMER_Frame.setInterval(30)
        self.TIMER_Frame.start()

        # 用于更新信号的定时器
        self.TIMER_SIGNAL = QTimer()
        self.TIMER_SIGNAL.setInterval(30)
        self.TIMER_SIGNAL.start()

        self.bpm_fore = 60
        self.bpm_left = 60
        self.bpm_right = 60

        self.bpm_avg = 60
        self.ModeDict = {'GREEN': self.processor.GREEN,
                         'GREEN-RED': self.processor.GREEN_RED, 'CHROM': self.processor.CHROM,
                         'PBV': self.processor.PBV}
        self.Mode = self.ModeDict['GREEN']
        self.Data_ShowRaw = True  # 展示原始信号或滤波信号
        self.slot_init()

    def slot_init(self):
        self.TIMER_Frame.timeout.connect(self.display_image_and_hist)
        self.TIMER_SIGNAL.timeout.connect(self.display_signal)
        self.comboBox.activated[str].connect(self.Button_ChangeMode)
        self.Button_RawTrue.clicked.connect(self.Button_Data_RawTrue)
        self.Button_RawFalse.clicked.connect(self.Button_Data_RawFalse)

    def Button_ChangeMode(self, str):
        self.Mode = self.ModeDict[str]

    def Button_Data_RawTrue(self):
        self.Data_ShowRaw = True

    def Button_Data_RawFalse(self):
        self.Data_ShowRaw = False

    def display_image_and_hist(self):
        """展示前置摄像头画面"""
        try:
            numbered_frame = self.series_class.masked_face_queue.get_frame()
        except Exception as e:
            print(e)
            return
        if numbered_frame is not None:
            masked_face = numbered_frame.masked_face
            img = cv.cvtColor(masked_face, cv.COLOR_BGR2RGB)
            qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.face.setPixmap(QPixmap.fromImage(qimg))
            self.display_hist(numbered_frame.hist_left, numbered_frame.hist_right, numbered_frame.hist_fore)

    def display_hist(self, hist_left, hist_right, hist_fore):
        """展示直方图数据"""
        def set_hist_data(hist, hist_r, hist_g, hist_b):
            if hist.size != 1:
                hist_r.setData(hist[0, :], pen=RED_PEN)
                hist_g.setData(hist[1, :], pen=GREEN_PEN)
                hist_b.setData(hist[2, :], pen=BLUE_PEN)
            else:
                hist_r.clear()
                hist_g.clear()
                hist_b.clear()
        hist_fore = np.array(hist_fore)
        hist_left = np.array(hist_left)
        hist_right = np.array(hist_right)
        set_hist_data(hist_fore, self.Hist_fore_r, self.Hist_fore_g, self.Hist_fore_b)
        set_hist_data(hist_left, self.Hist_left_r, self.Hist_left_g, self.Hist_left_b)
        set_hist_data(hist_right, self.Hist_right_r, self.Hist_right_g, self.Hist_right_b)

    # Creates the specified Butterworth filter and applies it.
    def butterworth_filter(self, data, low, high, sample_rate, order=11):
        """巴特沃斯滤波器"""
        nyquist_rate = sample_rate * 0.5
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)

    def process_signal(self, sig, bpm, sig_plot_widget, spec_plot_widget, pen):
        """计算信号并展示"""
        if sig.size != 1:
            bvp_raw = self.Mode(sig)    # 将信号转换为bvp(blood volume pulse)信号，属于PPG信号
            quality = 1 / (max(bvp_raw) - min(bvp_raw))
            bvp_filtered = self.butterworth_filter(
                self.processor.signal_preprocessing_single(bvp_raw), MIN_HZ, MAX_HZ,
                self.series_class.fps, order=5)
            spc = np.abs(np.fft.fft(bvp_filtered))  # 频域
            bpm = self.processor.cal_bpm(bpm, spc, self.series_class.fps)
            if self.Data_ShowRaw:
                sig_plot_widget.setData(bvp_raw, pen=pen)
            else:
                sig_plot_widget.setData(bvp_filtered, pen=pen)

            spec_plot_widget.setData(np.linspace(0, self.series_class.fps / 2 * ONE_MINUTE, int((len(spc) + 1) / 2)),
                                     spc[:int((len(spc) + 1) / 2)], pen=pen)
            return bvp_raw, quality, bvp_filtered, spc, bpm
        else:
            sig_plot_widget.setData([0], [0])
            spec_plot_widget.setData([0], [0])
            return None, None, None, None, bpm

    def display_signal(self):
        sig_fore = np.array(self.series_class.sig_fore)
        sig_left = np.array(self.series_class.sig_left)
        sig_right = np.array(self.series_class.sig_right)
        if self.series_class.data_collected:
            self.bvp_fore_raw, self.quality_fore, self.bvp_fore, self.spc_fore, self.bpm_fore = (
                self.process_signal(sig_fore, self.bpm_fore, self.Sig_f, self.Spec_f, (0, 255, 255)))

            self.bvp_left_raw, self.quality_left, self.bvp_left, self.spc_left, self.bpm_left = (
                self.process_signal(sig_left, self.bpm_left, self.Sig_l, self.Spec_l, (255, 0, 255)))

            self.bvp_right_raw, self.quality_right, self.bvp_right, self.spc_right, self.bpm_right = (
                self.process_signal(sig_right, self.bpm_right, self.Sig_r, self.Spec_r, (255, 255, 0)))

            self.quality_all = self.quality_fore + self.quality_left + self.quality_right
            if self.quality_all > 0:
                self.confidence_fore = self.quality_fore / self.quality_all
                self.confidence_left = self.quality_left / self.quality_all
                self.confidence_right = self.quality_right / self.quality_all
                self.bpm_avg = (self.bpm_fore * self.confidence_fore + self.bpm_left * self.confidence_left
                                + self.bpm_right * self.confidence_right)
            else:
                self.confidence_fore = 0
                self.confidence_left = 0
                self.confidence_right = 0
                self.bpm_avg = 60

            Label_Text = (
                f"Fps: \t\t{self.series_class.fps}\n"
                f"前额测量心率: \t{self.bpm_fore}\n"
                f"前额测量置信度: {self.confidence_fore * 100:.2f}%\n"
                f"左脸颊测量心率: \t{self.bpm_left}\n"
                f"左脸颊测量置信度: {self.confidence_left * 100:.2f}%\n"
                f"右脸颊测量心率:\t{self.bpm_right}\n"
                f"右脸颊测量置信度: {self.confidence_right * 100:.2f}%\n\n"
                f"最终心率: \t{self.bpm_avg:.2f}"
            )
            self.label.setText(Label_Text)
        else:
            self.Sig_f.setData([0], [0])
            self.Spec_f.setData([0], [0])
            self.Sig_l.setData([0], [0])
            self.Spec_l.setData([0], [0])
            self.Sig_r.setData([0], [0])
            self.Spec_r.setData([0], [0])
            self.label.setText(
                f"Fps:\t\t{self.series_class.fps}\n"
                f"收集数据中:\t\t {100 * self.series_class.get_process():.2f}%...")

    def closeEvent(self, a0):
        super().closeEvent(a0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainWin()
    ui.show()
    sys.exit(app.exec_())
