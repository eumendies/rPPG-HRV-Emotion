import logging
import os
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file

from hrv import ppg_hrv, analyze_emotion_hrv, calculate_emotion_scores
from plot import plot_emotion_and_ppg
from series2rPPG import array2ppg

app = Flask(__name__)

host = "localhost"
port = 6060

# 获取程序的运行目录
if getattr(sys, 'frozen', False):
    # 打包后的程序运行目录
    base_dir = os.path.dirname(sys.executable)
else:
    # 正常运行的程序目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

# 确保输出目录存在
output_dir = os.path.join(base_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


@app.route('/calculate_emotion', methods=['POST'])
def calculate_emotion():
    data = request.get_json()
    signal_array = np.array(data['signal'])

    try:
        bvp, bvp_filtered = array2ppg(signal_array, sampling_rate=30, mode="CHROM")
        quality = 1 / (np.max(bvp, axis=-1) - np.min(bvp, axis=-1))
        quality_all = np.sum(quality)
        confidence = quality / quality_all

        # 计算hrv
        hrv = []
        for i in range(3):  # 前额、左脸颊、右脸颊
            hrv.append(ppg_hrv(bvp_filtered[i], sampling_rate=30))
        hrv = pd.concat(hrv, axis=0)

        # 生成情绪报告与评分
        report, score = analyze_emotion_hrv(hrv)
        emotion_scores = [calculate_emotion_scores(hrv.iloc[[i]]) for i in range(len(hrv))]
        emotion_dict = {}
        for emotion in ['愤怒', '厌恶', '恐惧', '快乐', '悲伤', '惊讶']:
            emotion_dict[emotion] = (emotion_scores[0][emotion] * confidence[0]
                                     + emotion_scores[1][emotion] * confidence[1]
                                     + emotion_scores[2][emotion] * confidence[2])

        # 绘图
        file_id = uuid4().hex
        filename = f"{file_id}.png"
        #output_path = f"./output/{file_id}.png"
        output_path = os.path.join(output_dir, filename)
        get_path = f"http://{host}:{port}/image/{file_id}.png"
        plot_emotion_and_ppg(emotion_dict, bvp_filtered, output_path=output_path)

        result = {
            "report": report,
            "score": score,
            "image": get_path
        }

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/image/<path:filename>')
def get_image(filename):
    # path = f"./output/{filename}"
    path = os.path.join(output_dir, filename)
    try:
        if not os.path.exists(path):
            return {'error': 'File not found'}, 404

        return send_file(path)
    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == '__main__':
    app.logger.setLevel(logging.INFO)
    app.run(host=host, port=port, debug=True)
