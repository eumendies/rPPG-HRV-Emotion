"""
Author: Harryhht
Date: 2022-02-01 20:51:25
LastEditors: Eumendies
LastEditTime: 2025-04-10 14:30:50
Description: Main structure for the application
"""

import sys
import time

import cv2 as cv
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy import signal

from constants import RED_PEN, GREEN_PEN, BLUE_PEN, ONE_MINUTE
from face2series import CAM2FACE
from hrv import ppg_hrv
from mainwindow import Ui_MainWindow
from series2rPPG import Series2rPPG

MIN_HZ = 0.83  # 50 BPM - minimum allowed heart rate
MAX_HZ = 2.5  # 150 BPM - maximum allowed heart rate


class MainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWin, self).__init__(parent)
        self.setup_ui(self)

        self.processor = Series2rPPG()
        self.series_class = CAM2FACE()

        self.hrv_calculation_interval = 5   # 每5秒计算一次HRV
        self.last_calculation_time = 0

        self.bpm_fore = 60
        self.bpm_left = 60
        self.bpm_right = 60

        self.bpm_avg = 60
        self.Mode = 'GREEN'
        self.Data_ShowRaw = True  # 展示原始信号或滤波信号
        self.slot_init()

    def slot_init(self):
        self.comboBox.activated[str].connect(self.combobox_change_mode)
        self.comboBox_data_num.activated[str].connect(self.combobox_change_data_num)
        self.Button_Raw.clicked.connect(self.button_data_raw_true)
        self.Button_Filtered.clicked.connect(self.button_data_raw_false)

        self.series_class.image_signal.connect(self.display_image_and_hist)
        self.series_class.features_signal.connect(self.display_signal)
        self.series_class.start()

    def combobox_change_mode(self, str):
        self.Mode = str

    def button_data_raw_true(self):
        self.Data_ShowRaw = True

    def button_data_raw_false(self):
        self.Data_ShowRaw = False

    def combobox_change_data_num(self, data_num):
        if data_num == '采集数据量(默认256)':
            data_num = 256
        if data_num != self.series_class.QUEUE_MAX:
            self.series_class.change_data_num(int(data_num))
        self.Sig_f.setData([0], [0])
        self.Spec_f.setData([0], [0])
        self.Sig_l.setData([0], [0])
        self.Spec_l.setData([0], [0])
        self.Sig_r.setData([0], [0])
        self.Spec_r.setData([0], [0])

    def display_image_and_hist(self, numbered_frame):
        """展示前置摄像头画面"""
        if numbered_frame is not None:
            masked_face = numbered_frame.masked_face
            img = cv.cvtColor(masked_face, cv.COLOR_BGR2RGB)
            qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.face.setPixmap(QPixmap.fromImage(qimg))
            self.display_hist(numbered_frame.hist_left, numbered_frame.hist_right, numbered_frame.hist_fore)
        if self.series_class.get_process() < 1:
            self.info_label.setText(f"Fps: \t\t{self.series_class.fps:.2f}\n"
                                    f"收集数据中: \t\t{100 * self.series_class.get_process():.2f}%")


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

    def calc_bvp(self, signal):
        """将信号转换为bvp(blood volume pulse)信号，属于PPG信号"""
        if self.Mode == 'GREEN':
            return self.processor.GREEN(signal)
        elif self.Mode == 'GREEN-RED':
            return self.processor.GREEN_RED(signal)
        elif self.Mode == 'CHROM':
            return self.processor.CHROM(signal)
        elif self.Mode == 'PBV':
            return self.processor.PBV(signal)
        elif self.Mode == 'POS':
            return self.processor.POS(signal, self.series_class.fps)

    def display_bpm(self, bpm_dict):
        """将各ROI的测量心率和置信度展示在表格中"""
        for i, header in enumerate(["前额测量心率", "左脸颊测量心率", "右脸颊测量心率", "最终心率"]):
            value_item = QTableWidgetItem(str(round(bpm_dict[header][0], 2)))
            confidence_item = QTableWidgetItem(str(round(bpm_dict[header][1], 2)))
            value_item.setTextAlignment(Qt.AlignCenter)
            confidence_item.setTextAlignment(Qt.AlignCenter)
            self.info_table.setItem(0, i, value_item)
            self.info_table.setItem(1, i, confidence_item)

    def calc_hrv(self, bvp, row):
        """计算HRV各项指标并显示在表格中"""
        if bvp.size != 1:
            hrv = ppg_hrv(bvp, self.series_class.fps)
            for col in range(self.time_hrv_table.columnCount()):
                header = self.time_hrv_table.horizontalHeaderItem(col).text()
                item = QTableWidgetItem(str(round(hrv[header].item(), 2)))
                item.setTextAlignment(Qt.AlignCenter)
                self.time_hrv_table.setItem(row, col, item)
            for col in range(self.freq_hrv_table.columnCount()):
                header = self.freq_hrv_table.horizontalHeaderItem(col).text()
                item = QTableWidgetItem(str(round(hrv[header].item(), 2)))
                item.setTextAlignment(Qt.AlignCenter)
                self.freq_hrv_table.setItem(row, col, item)

    def plot_bvp_and_spec(self, bvp_raw, bvp_filtered, spec, sig_plot_widget, spec_plot_widget, pen):
        if self.Data_ShowRaw:
            sig_plot_widget.setData(bvp_raw, pen=pen)
        else:
            sig_plot_widget.setData(bvp_filtered, pen=pen)

        spec_plot_widget.setData(np.linspace(0, self.series_class.fps / 2 * ONE_MINUTE, int((len(spec) + 1) / 2)),
                                 spec[:int((len(spec) + 1) / 2)], pen=pen)

    def display_signal(self, signal):
        """
        args:
            signal: shape [3, len(queue), 3]
        """
        bvp = self.calc_bvp(signal)
        bvp_raw = bvp[:, -self.series_class.FEATURE_WINDOW:]  # [3, FEATURE_WINDOW]
        quality = 1 / (np.max(bvp_raw, axis=-1) - np.min(bvp_raw, axis=-1))
        bvp_filtered = np.array([self.butterworth_filter(self.processor.signal_preprocessing_single(bvp_raw[i, :]),
                                                         MIN_HZ, MAX_HZ, self.series_class.fps, order=5)
                                 for i in range(3)])
        spec = np.abs(np.fft.fft(bvp_filtered))

        self.bpm_fore = self.processor.cal_bpm(self.bpm_fore, spec[0, :], self.series_class.fps)
        self.bpm_left = self.processor.cal_bpm(self.bpm_left, spec[1, :], self.series_class.fps)
        self.bpm_right = self.processor.cal_bpm(self.bpm_right, spec[2, :], self.series_class.fps)

        self.plot_bvp_and_spec(bvp_raw[0, :], bvp_filtered[0, :], spec[0, :], self.Sig_f, self.Spec_f, (0, 255, 255))
        self.plot_bvp_and_spec(bvp_raw[1, :], bvp_filtered[1, :], spec[1, :], self.Sig_l, self.Spec_l, (255, 0, 255))
        self.plot_bvp_and_spec(bvp_raw[2, :], bvp_filtered[2, :], spec[2, :], self.Sig_r, self.Spec_r, (255, 255, 0))

        if time.time() - self.last_calculation_time > self.hrv_calculation_interval:
            self.calc_hrv(bvp[0, :], 0)
            self.calc_hrv(bvp[1, :], 1)
            self.calc_hrv(bvp[2, :], 2)
            self.last_calculation_time = time.time()

        quality_all = np.sum(quality)
        if quality_all > 0:
            confidence = quality / quality_all
            self.bpm_avg = np.matmul(np.array([self.bpm_fore, self.bpm_left, self.bpm_right]), confidence.transpose())
        else:
            confidence = np.array([0, 0, 0])
            self.bpm_avg = 60

        bpm_dict = {
            "前额测量心率": [self.bpm_fore, confidence[0]],
            "左脸颊测量心率": [self.bpm_left, confidence[1]],
            "右脸颊测量心率": [self.bpm_right, confidence[2]],
            "最终心率": [self.bpm_avg, 1]
        }
        self.display_bpm(bpm_dict)
        self.info_label.setText(f"Fps: \t\t{self.series_class.fps:.2f}")

    def closeEvent(self, a0):
        super().closeEvent(a0)
        self.series_class.__del__()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainWin()
    ui.show()
    sys.exit(app.exec_())
