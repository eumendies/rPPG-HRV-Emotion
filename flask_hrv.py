from flask import Flask, request, jsonify
import numpy as np
from hrv import ppg_hrv, estimate_emotions
from series2rPPG import Series2rPPG
from scipy import signal

app = Flask(__name__)

# 初始化必要组件
processor = Series2rPPG()
fps = 30
MIN_HZ = 0.83  # 50 BPM - minimum allowed heart rate
MAX_HZ = 2.5  # 150 BPM - maximum allowed heart rate


def butterworth_filter(data, sample_rate, low=MIN_HZ, high=MAX_HZ, order=11):
    """巴特沃斯滤波器"""
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


def calc_bvp(signal):
    return processor.CHROM(signal)


@app.route('/calculate_emotion', methods=['POST'])
def calculate_emotion():
    data = request.get_json()
    signal_array = np.array(data['signal'])

    try:
        bvp = calc_bvp(signal_array)
        bvp_filtered = filter_bvp(bvp)
        quality = 1 / (np.max(bvp, axis=-1) - np.min(bvp, axis=-1))

        emotion_results = []
        # 对每个区域计算HRV
        for i in range(3):  # 前额、左脸颊、右脸颊
            hrv_data = ppg_hrv(bvp_filtered[i], fps)
            emotion_results.append(estimate_emotions(hrv_data))

        quality_all = np.sum(quality)
        if quality_all > 0:
            confidence = quality / quality_all
        else:
            confidence = np.array([0, 0, 0])

        result = {}
        for emotion in emotion_results[0]:
            result[emotion] = (emotion_results[0][emotion] * confidence[0]
                               + emotion_results[1][emotion] * confidence[1]
                               + emotion_results[2][emotion] * confidence[2])

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def filter_bvp(bvp):
    """信号过滤处理"""
    filtered = []
    for i in range(3):
        filtered.append(
            butterworth_filter(
                processor.signal_preprocessing_single(bvp[i]),
                sample_rate=fps,
                order=5
            )
        )
    return np.array(filtered)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5060, debug=True)
