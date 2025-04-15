import copy
import queue
import threading
import time
from queue import Queue

import cv2 as cv
import dlib
import numpy as np
import seaborn as sns

sns.set()


class NumberedFrame:
    """带序号的帧数据"""

    def __init__(self, frame, frame_count):
        self.frame = frame
        self.frame_count = frame_count
        self.masked_face = None
        self.hist_left = None
        self.hist_right = None
        self.hist_fore = None

    def set_hist(self, hist_left, hist_right, hist_fore):
        self.hist_left = hist_left
        self.hist_right = hist_right
        self.hist_fore = hist_fore


class FrameQueue:
    """帧序列，其中的帧按时间戳排序，每次取出时间戳最小的"""

    def __init__(self, maxsize=0):
        self.queue = queue.PriorityQueue(maxsize=maxsize)
        self.allow_frame_count = 0  # 最后被取出的帧的序号，如果新put进来的帧小于该号，则直接丢弃
        self.mutex = threading.Lock()

    def get_frame(self) -> NumberedFrame | None:
        if self.queue.empty():
            return None
        with self.mutex:
            # 保证取帧和给allow_frame_count赋值两个操作称为原子操作
            _, frame = self.queue.get()
            self.allow_frame_count = frame.frame_count
            return frame

    def put_frame(self, frame: NumberedFrame):
        with self.mutex:
            if self.allow_frame_count < frame.frame_count:
                self.queue.put((frame.frame_count, frame))


class CAM2FACE:
    """负责读取摄像头、识别三个ROI（左右脸颊和额头）、将RGB值转换为特征"""

    def __init__(self, num_process_threads=4) -> None:
        # get face detector and 68 face landmark
        self.num_process_threads = num_process_threads
        self.detectors = [dlib.get_frontal_face_detector() for _ in range(num_process_threads)]
        self.predictors = [dlib.shape_predictor('data/shape_predictor_81_face_landmarks.dat') for _ in
                           range(num_process_threads)]

        # get frontal camera of computer and get fps
        self.cam = cv.VideoCapture(0)
        if not self.cam.isOpened():
            print('ERROR:  Unable to open webcam.  Verify that webcam is connected and try again.  Exiting.')
            self.cam.release()
            return
        self.fps = self.cam.get(cv.CAP_PROP_FPS)

        # Initialize Queue for camera capture
        self.QUEUE_MAX = 256
        self.QUEUE_WINDOWS = 64
        self.queue_rawframe = Queue()
        self.queue_sig_left = queue.PriorityQueue(maxsize=self.QUEUE_MAX)  # 左脸颊信号队列
        self.queue_sig_right = queue.PriorityQueue(maxsize=self.QUEUE_MAX)  # 右脸颊信号队列
        self.queue_sig_fore = queue.PriorityQueue(maxsize=self.QUEUE_MAX)  # 额头信号队列

        self.queue_time = Queue(maxsize=self.QUEUE_WINDOWS)
        self.mutex = threading.Lock()

        self.ongoing = False
        self.data_collected = False  # 队列是否已满，已满后才可视化信号

        # 多线程处理，使用优先队列保证处理后的帧有序
        # TODO: 实际上仍可能有问题，比如线程1处理第1帧，线程2处理第2帧，
        #  线程2连续处理完5帧后线程1才处理完第1帧，此时线程2的5帧可能已经被播放了，目前把迟到的帧丢弃
        self.masked_face_queue = FrameQueue()

        self.sig_left = None
        self.sig_right = None
        self.sig_fore = None

    # Initialize process and start

    def PROCESS_start(self):
        self.ongoing = True
        self.capture_thread = threading.Thread(target=self.capture_process)
        self.roi_cal_threads = [threading.Thread(target=self.roi_cal_process, args=(i,), daemon=True) for i in
                                range(self.num_process_threads)]

        self.capture_thread.start()
        for thread in self.roi_cal_threads:
            thread.start()

    # Process: capture frame from camera in specific fps of the camera
    def capture_process(self):
        while self.ongoing:
            ret, frame = self.cam.read()
            if not ret:
                self.ongoing = False
                break
            self.queue_rawframe.put(NumberedFrame(frame, time.time()))

    # Process: calculate roi from raw frame
    def roi_cal_process(self, thread_id):
        while self.ongoing:
            numbered_frame = self.queue_rawframe.get()
            frame_count = numbered_frame.frame_count
            # get the roi of the frame (left/right)
            detector = self.detectors[thread_id]
            predictor = self.predictors[thread_id]
            masked_face, roi_left, roi_right, roi_fore = self.ROI(numbered_frame, detector, predictor)
            if roi_left is not None and roi_right is not None and roi_fore is not None:
                # produce rgb hist of mask (removed black)
                hist_left = self.rgb_hist(roi_left)
                hist_right = self.rgb_hist(roi_right)
                hist_fore = self.rgb_hist(roi_fore)

                numbered_frame.masked_face = masked_face
                numbered_frame.set_hist(hist_left, hist_right, hist_fore)
                self.masked_face_queue.put_frame(numbered_frame)
                with self.mutex:
                    self.data_collected = self.queue_sig_fore.full()
                    if self.queue_sig_left.full():
                        self.sig_left = [value for _, value in copy.copy(self.queue_sig_left.queue)]
                        self.queue_sig_left.get_nowait()
                    if self.queue_sig_right.full():
                        self.sig_right = [value for _, value in copy.copy(self.queue_sig_right.queue)]
                        self.queue_sig_right.get_nowait()
                    if self.queue_sig_fore.full():
                        self.sig_fore = [value for _, value in copy.copy(self.queue_sig_fore.queue)]
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
                print("No face detected")
                self.queue_sig_left.queue.clear()
                self.queue_sig_right.queue.clear()
                self.queue_sig_fore.queue.clear()

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
        return cv.GaussianBlur(img, (5, 5), 0)

    # Draw the ROI the image
    # ROI: left cheek and right cheek
    def ROI(self, numbered_frame: NumberedFrame, detector, predictor):
        img = numbered_frame.frame
        img = cv.resize(img, (1080, 720))  # TODO: 缩放图像减小计算量，需要先判断摄像头分辨率
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

        # sgn_r = np.tanh(hist_r)
        # sgn_g = np.tanh(hist_g)
        # sgn_b = np.tanh(hist_b)

        hist_r /= np.sum(hist_r)
        hist_g /= np.sum(hist_g)
        hist_b /= np.sum(hist_b)

        dens = np.arange(0, 256, 1)
        mean_r = dens.dot(hist_r)
        mean_g = dens.dot(hist_g)
        mean_b = dens.dot(hist_b)

        return [mean_r, mean_g, mean_b]

    # Deconstruction

    def __del__(self):
        self.ongoing = False
        self.cam.release()
        cv.destroyAllWindows()

    def get_process(self):
        """收集数据进度"""
        return self.queue_sig_fore.qsize() / self.QUEUE_MAX


if __name__ == '__main__':
    cam2roi = CAM2FACE()
    cam2roi.PROCESS_start()
    Hist_left_list = []
    Hist_right_list = []
    while True:
        print(cam2roi.fps)
    # time.sleep(1)
    # while True:
    # Hist_left = cam2roi.Queue_RGBhist_left.get()
    # Hist_right = cam2roi.Queue_RGBhist_right.get()
    # print(Hist_left)
    # cam2roi.__del__()
