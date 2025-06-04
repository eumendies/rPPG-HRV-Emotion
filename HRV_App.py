import sys
import os
import threading
import logging
from PyQt5.QtWidgets import QApplication

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入Flask应用
from flask_hrv import app, host, port

# 导入PyQt5应用
from hrv_main import DetectionWindow, LoginWindow, CAM2FACE


def run_flask():
    """运行Flask服务器"""
    logging.info(f"Starting Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


def run_gui():
    """运行PyQt5 GUI"""
    logging.info("Starting PyQt5 application")
    series_class = CAM2FACE()
    qt_app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()

    detection_window = DetectionWindow(series_class=series_class, crop_size=250)
    login_window.close_signal.connect(detection_window.show)
    login_window.login_signal.connect(detection_window.set_student_id)

    sys.exit(qt_app.exec_())


if __name__ == '__main__':
    # 启动Flask线程
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # 在主线程运行GUI
    run_gui()