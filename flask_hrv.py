import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from plot import plot_ppg_signal

from hrv import ppg_hrv, analyze_emotion_hrv
from series2rPPG import array2ppg
from uuid import uuid4

app = Flask(__name__)

host = "localhost"
port = 6060


@app.route('/calculate_emotion', methods=['POST'])
def calculate_emotion():
    data = request.get_json()
    signal_array = np.array(data['signal'])

    try:
        bvp, bvp_filtered = array2ppg(signal_array, sampling_rate=30, mode="CHROM")

        file_id = uuid4().hex
        output_path = f"./output/{file_id}.png"
        get_path = f"http://{host}:{port}/image/{file_id}.png"
        plot_ppg_signal(bvp_filtered, output_path=output_path)

        hrv = []
        for i in range(3):  # 前额、左脸颊、右脸颊
            hrv.append(ppg_hrv(bvp_filtered[i], sampling_rate=30))
        hrv = pd.concat(hrv, axis=0)
        report, score = analyze_emotion_hrv(hrv)

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
    path = f"./output/{filename}"
    try:
        if not os.path.exists(path):
            return {'error': 'File not found'}, 404

        return send_file(path)
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)
