import queue
import threading


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


class OrderedFrameQueue:
    """帧序列，其中的帧按时间戳排序，每次取出时间戳最小的，迟到的帧直接丢弃"""

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