import copy
import os
import queue
import sys
import threading
import time
from queue import Queue

import cv2 as cv
import dlib
import numpy as np
import seaborn as sns
from PyQt5.QtCore import QThread, pyqtSignal
from constants import ONE_HOUR
from entities import OrderedFrameQueue, NumberedFrame

sns.set()


def count_cameras():
    """检测系统中有多少个可用的摄像头"""
    max_cameras = 10  # 设置最大尝试数量
    camera_count = 0

    for index in range(max_cameras):
        cap = cv.VideoCapture(index)
        if cap.isOpened():
            camera_count += 1
            cap.release()
        else:
            break
    return camera_count


def get_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.normpath(os.path.join(base_path, relative_path))


class CAM2FACE(QThread):
    """负责读取摄像头、识别三个ROI（左右脸颊和额头）、将RGB值转换为特征"""
    image_signal = pyqtSignal(object)  # 发送处理后的图像
    features_signal = pyqtSignal(object)
    detected_signal = pyqtSignal(bool)

    def __init__(self, num_process_threads=4) -> None:
        super().__init__()
        self.num_process_threads = num_process_threads
        self.detectors = [dlib.get_frontal_face_detector() for _ in range(num_process_threads)]
        self.predictors = [dlib.shape_predictor(get_path('data/shape_predictor_81_face_landmarks.dat')) for _ in
                           range(num_process_threads)]

        # get frontal camera of computer and get fps
        self.total_camera_count = count_cameras()
        self.current_camera = 0
        self.cam = cv.VideoCapture(self.current_camera)
        if not self.cam.isOpened():
            print('ERROR:  Unable to open webcam.  Verify that webcam is connected and try again.  Exiting.')
            self.cam.release()
            return
        self.fps = self.cam.get(cv.CAP_PROP_FPS)

        self.QUEUE_MAX = ONE_HOUR * self.fps  # 最多保存一小时内的数据

        self.FEATURE_WINDOW = 256  # 用于展示可视化图的数据窗口大小
        self.QUEUE_WINDOWS = 64
        self.queue_rawframe = Queue()
        self.queue_sig_left = queue.PriorityQueue(maxsize=self.QUEUE_MAX)  # 左脸颊信号队列
        self.queue_sig_right = queue.PriorityQueue(maxsize=self.QUEUE_MAX)  # 右脸颊信号队列
        self.queue_sig_fore = queue.PriorityQueue(maxsize=self.QUEUE_MAX)  # 额头信号队列

        self.queue_time = Queue(maxsize=self.QUEUE_WINDOWS)
        self.mutex = threading.Lock()
        self.until_stable = False  # 等到检测稳定后再将计算出来的特征emit，抛弃掉未稳定时计算的特征
        self.stable_period = 128  # 抛弃128个数据

        self.ongoing = False

        # 多线程处理，使用优先队列保证处理后的帧有序
        self.masked_face_queue = OrderedFrameQueue()

    def run(self):
        self.ongoing = True

        self.capture_thread = threading.Thread(target=self.capture_process)
        self.roi_cal_threads = [threading.Thread(target=self.roi_cal_process, args=(i,), daemon=True) for i in
                                range(self.num_process_threads)]
        self.capture_thread.start()
        for thread in self.roi_cal_threads:
            thread.start()

        while self.ongoing:
            if (not self.until_stable
                    and self.queue_sig_fore.qsize() > self.stable_period
                    and self.queue_sig_right.qsize() > self.stable_period
                    and self.queue_sig_left.qsize() > self.stable_period):
                self.queue_sig_fore.queue.clear()
                self.queue_sig_left.queue.clear()
                self.queue_sig_right.queue.clear()
                self.until_stable = True

            if (self.queue_sig_fore.qsize() >= self.FEATURE_WINDOW
                    and self.queue_sig_right.qsize() >= self.FEATURE_WINDOW
                    and self.queue_sig_fore.qsize() >= self.FEATURE_WINDOW):
                fore_features = [value for _, value in copy.copy(self.queue_sig_fore.queue)]
                left_features = [value for _, value in copy.copy(self.queue_sig_left.queue)]
                right_features = [value for _, value in copy.copy(self.queue_sig_right.queue)]
                min_len = min(len(fore_features), len(left_features), len(right_features))
                features = np.array(
                    [fore_features[:min_len], left_features[:min_len], right_features[:min_len]])  # [3, len(queue), 3]
                self.features_signal.emit(features)
            self.msleep(20)

    def stop(self):
        self.ongoing = False
        self.until_stable = False
        self.clear_queue()

    def clear_queue(self):
        self.queue_rawframe.queue.clear()
        self.queue_sig_right.queue.clear()
        self.queue_sig_fore.queue.clear()
        self.queue_sig_left.queue.clear()
        self.queue_time.queue.clear()
        self.masked_face_queue.reset()

    def change_data_num(self, data_num):
        self.FEATURE_WINDOW = data_num

    def capture_process(self):
        while self.ongoing:
            ret, frame = self.cam.read()
            if not ret:
                self.ongoing = False
                break
            numbered_frame = NumberedFrame(frame, time.time())
            self.queue_rawframe.put(numbered_frame)
            self.image_signal.emit(numbered_frame)

    def roi_cal_process(self, thread_id):
        while self.ongoing:
            numbered_frame = self.queue_rawframe.get()
            frame_count = numbered_frame.frame_count
            # get the roi of the frame (left/right)
            detector = self.detectors[thread_id]
            predictor = self.predictors[thread_id]
            masked_face, roi_left, roi_right, roi_fore = self.ROI(numbered_frame, detector, predictor)
            if roi_left is not None and roi_right is not None and roi_fore is not None:
                self.detected_signal.emit(True)
                # produce rgb hist of mask (removed black)
                hist_left = self.rgb_hist(roi_left)
                hist_right = self.rgb_hist(roi_right)
                hist_fore = self.rgb_hist(roi_fore)

                numbered_frame.masked_face = masked_face
                numbered_frame.set_hist(hist_left, hist_right, hist_fore)
                self.masked_face_queue.put_frame(numbered_frame)
                with self.mutex:
                    if self.queue_sig_left.full():
                        self.queue_sig_left.get_nowait()
                    if self.queue_sig_right.full():
                        self.queue_sig_right.get_nowait()
                    if self.queue_sig_fore.full():
                        self.queue_sig_fore.get_nowait()

                    self.queue_sig_left.put_nowait((frame_count, self.hist2feature(hist_left)))
                    self.queue_sig_right.put_nowait((frame_count, self.hist2feature(hist_right)))
                    self.queue_sig_fore.put_nowait((frame_count, self.hist2feature(hist_fore)))

                    # 计算处理帧率
                    if self.queue_time.full():
                        self.queue_time.get_nowait()
                    self.queue_time.put_nowait(time.time())
                    if self.queue_time.full():
                        self.fps = 1 / np.mean(np.diff(np.array(list(self.queue_time.queue))))  # 时间的一阶差分的平均值的倒数为fps
            else:
                self.until_stable = False
                self.queue_sig_left.queue.clear()
                self.queue_sig_right.queue.clear()
                self.queue_sig_fore.queue.clear()
                self.detected_signal.emit(False)

    def detect_landmarks(self, img, detector, predictor):
        """获取脸部关键点"""
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        if len(faces) == 1:
            face = faces[0]
            landmarks = [[p.x, p.y] for p in predictor(img, face).parts()]
            return landmarks
        return None

    # filter the image to ensure better performance
    def preprocess(self, img):
        img = cv.resize(img, (1080, 720))  # TODO: 缩放图像减小计算量，需要先判断摄像头分辨率
        # return cv.GaussianBlur(img, (5, 5), 0)
        return img

    # Draw the ROI the image
    # ROI: left cheek and right cheek
    def ROI(self, numbered_frame: NumberedFrame, detector, predictor):
        img = numbered_frame.frame
        img = self.preprocess(img)
        landmark = self.detect_landmarks(img, detector, predictor)

        if landmark is None:
            return None, None, None, None

        # 左脸颊、右脸颊和额头区域的关键点的索引
        cheek_left = [1, 2, 3, 4, 48, 31, 28, 39]
        cheek_right = [15, 14, 14, 12, 54, 35, 28, 42]
        forehead = [69, 70, 71, 80, 72, 25, 24, 23, 22, 21, 20, 19, 18]

        mask_left = np.zeros(img.shape, np.uint8)
        mask_right = np.zeros(img.shape, np.uint8)
        mask_fore = np.zeros(img.shape, np.uint8)
        try:
            pts_left = np.array([landmark[i] for i in cheek_left], np.int32).reshape((-1, 1, 2))
            pts_right = np.array([landmark[i] for i in cheek_right], np.int32).reshape((-1, 1, 2))
            pts_fore = np.array([landmark[i] for i in forehead], np.int32).reshape((-1, 1, 2))
            mask_left = cv.fillPoly(mask_left, [pts_left], (255, 255, 255))
            mask_right = cv.fillPoly(mask_right, [pts_right], (255, 255, 255))
            mask_fore = cv.fillPoly(mask_fore, [pts_fore], (255, 255, 255))

            # Erode Kernel: 30
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 30))
            mask_left = cv.erode(mask_left, kernel=kernel, iterations=1)
            mask_right = cv.erode(mask_right, kernel=kernel, iterations=1)
            mask_fore = cv.erode(mask_fore, kernel=kernel, iterations=1)

            mask_display_left, mask_display_right, mask_display_fore = (
                copy.copy(mask_left), copy.copy(mask_right), copy.copy(mask_fore))
            # 将某个通道置为0，从而将其颜色从白色改为其他颜色
            mask_display_left[:, :, 1] = 0
            mask_display_right[:, :, 0] = 0
            mask_display_fore[:, :, 2] = 0

            mask_display = cv.bitwise_or(mask_display_left, mask_display_right)
            mask_display = cv.bitwise_or(mask_display, mask_display_fore)

            # 将掩膜和原图进行混合
            masked_face = cv.addWeighted(mask_display, 0.25, img, 1, 0)
            ROI_left = cv.bitwise_and(mask_left, img)
            ROI_right = cv.bitwise_and(mask_right, img)
            ROI_fore = cv.bitwise_and(mask_fore, img)
            return masked_face, ROI_left, ROI_right, ROI_fore

        except Exception as e:
            print(e)
            return None, None, None, None

    # Cal hist of roi
    def rgb_hist(self, roi):
        b_hist = cv.calcHist([roi], [0], None, [256], [0, 256])
        g_hist = cv.calcHist([roi], [1], None, [256], [0, 256])
        r_hist = cv.calcHist([roi], [2], None, [256], [0, 256])
        b_hist = np.reshape(b_hist, (256))
        g_hist = np.reshape(g_hist, (256))
        r_hist = np.reshape(r_hist, (256))
        b_hist[0] = 0
        g_hist[0] = 0
        r_hist[0] = 0
        r_hist = r_hist / np.sum(r_hist)
        g_hist = g_hist / np.sum(g_hist)
        b_hist = b_hist / np.sum(b_hist)
        return [r_hist, g_hist, b_hist]

    def hist2feature(self, hist):
        """从RGB直方图中提取特征，即RGB均值"""
        hist_r = hist[0]
        hist_g = hist[1]
        hist_b = hist[2]

        hist_r /= np.sum(hist_r)
        hist_g /= np.sum(hist_g)
        hist_b /= np.sum(hist_b)

        dens = np.arange(0, 256, 1)
        mean_r = dens.dot(hist_r)
        mean_g = dens.dot(hist_g)
        mean_b = dens.dot(hist_b)
        return [mean_r, mean_g, mean_b]

    def get_signals(self):
        if (self.queue_sig_fore.qsize() >= self.FEATURE_WINDOW
                and self.queue_sig_right.qsize() >= self.FEATURE_WINDOW
                and self.queue_sig_fore.qsize() >= self.FEATURE_WINDOW):
            fore_features = [value for _, value in copy.copy(self.queue_sig_fore.queue)]
            left_features = [value for _, value in copy.copy(self.queue_sig_left.queue)]
            right_features = [value for _, value in copy.copy(self.queue_sig_right.queue)]
            min_len = min(len(fore_features), len(left_features), len(right_features))
            features = np.array(
                [fore_features[:min_len], left_features[:min_len], right_features[:min_len]])  # [3, len(queue), 3]
            return features
        return None

    def __del__(self):
        self.ongoing = False
        self.cam.release()
        # cv.destroyAllWindows()

    def get_progress(self):
        """收集数据进度"""
        if not self.until_stable:
            return 0
        return min(self.queue_sig_fore.qsize() / self.FEATURE_WINDOW,
                   self.queue_sig_left.qsize() / self.FEATURE_WINDOW,
                   self.queue_sig_right.qsize() / self.FEATURE_WINDOW)

    def switch_camera(self):
        next_camera = (self.current_camera + 1) % self.total_camera_count
        if next_camera == self.current_camera:
            return

        self.ongoing = False
        # 等待capture线程结束
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        # 释放当前摄像头
        if hasattr(self, 'cam') and self.cam.isOpened():
            self.cam.release()
        self.current_camera = next_camera
        self.cam = cv.VideoCapture(self.current_camera)
        self.clear_queue()
        self.start()

