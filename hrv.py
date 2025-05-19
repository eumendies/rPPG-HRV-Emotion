import json

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from series2rPPG import array2ppg
from plot import plot_ppg_signal


def ppg_hrv(ppg_signal, sampling_rate):
    """使用ppg信号计算hrv"""
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)
    ppg_peaks, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
    df_time = nk.hrv_time(ppg_peaks, sampling_rate=sampling_rate, show=False)
    df_nonlinear = nk.hrv_nonlinear(ppg_peaks, sampling_rate=sampling_rate, show=False)
    df_freq = nk.hrv_frequency(ppg_peaks, sampling_rate=sampling_rate, show=False)
    return pd.concat([df_time, df_freq, df_nonlinear], axis=1)


def estimate_emotions(hrv_data):
    """
    基于HRV时域指标估算情绪状态

    参数:
    hrv_data -- 包含HRV时域指标的DataFrame，只有一行，包含以下列:
                "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_pNN50"

    返回:
    包含六种情绪得分的字典，每个情绪得分为0-1之间的值，总和为1
    """
    mean_nn = hrv_data["HRV_MeanNN"].values[0]
    sdnn = hrv_data["HRV_SDNN"].values[0]
    rmssd = hrv_data["HRV_RMSSD"].values[0]
    sdsd = hrv_data["HRV_SDSD"].values[0]
    pnn50 = hrv_data["HRV_pNN50"].values[0]

    emotions = {}
    emotions["愤怒"] = 0.5 / sdnn + 0.3 / (pnn50 + 0.1) + 0.2 / (rmssd + 0.1)
    emotions["厌恶"] = 0.4 / (sdnn + 0.1) + 0.3 / (rmssd + 0.1) + 0.3 / (pnn50 + 0.1)
    emotions["恐惧"] = 0.6 / (sdnn + 0.01) + 0.3 / (rmssd + 0.01) + 0.1 / (pnn50 + 0.01)
    emotions["快乐"] = 0.5 * np.log(sdnn + 1) + 0.3 * np.log(rmssd + 1) + 0.2 * np.log(pnn50 + 1)
    emotions["悲伤"] = 0.2 * np.log(1 / (sdnn + 0.1)) + 0.3 * np.log(1 / (rmssd + 0.1)) + 0.5 * np.log(
        1 / (pnn50 + 0.1))
    emotions["惊讶"] = 0.4 * np.log(1 / (sdsd + 0.1)) + 0.3 * np.log(1 / (rmssd + 0.1)) + 0.3 * np.log(
        1 / (pnn50 + 0.1))

    sum_scores = sum(emotions.values())
    for emotion in emotions:
        emotions[emotion] = emotions[emotion] / sum_scores

    return emotions


def train_and_save_models(df, features, output_path):
    """
    训练无监督聚类模型并保存

    参数:
        df (DataFrame): 包含HRV时域特征的DataFrame
        features (list): 要使用的特征列名
        output_path (str): 模型保存的路径前缀

    返回:
        dict: 包含训练好的模型对象的字典
    """
    feature_data = df[features].values

    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_data)

    # 训练各种聚类模型
    models = {
        'KMeans': KMeans(n_clusters=2, random_state=42),
        'GMM': GaussianMixture(n_components=2, random_state=42)
    }
    for name, model in models.items():
        model.fit(feature_scaled)
        joblib.dump((scaler, model), f"{output_path}_{name}.joblib")
    return models


def load_model_and_predict(model_path, new_data):
    """
    加载训练好的模型并进行预测

    参数:
        model_path (str): 模型文件路径
        new_data (DataFrame): 包含待预测数据的DataFrame

    返回:
        int: 预测的聚类标签
    """
    scaler, model = joblib.load(model_path)
    features = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_pNN50"]

    # 需要至少2行数据才能使用scaler.transform
    if len(new_data) == 1:
        new_data = pd.concat([new_data, new_data], ignore_index=True)

    X_new_scaled = scaler.transform(new_data[features])

    if isinstance(model, DBSCAN):
        return model.fit_predict(X_new_scaled)
    else:
        return model.predict(X_new_scaled)


def analyze_emotion_hrv(df):
    """
    输入：包含HRV参数的DataFrame（单行数据）
    输出：情绪报告文本，心理评分（0-100）
    """
    # 提取参数值（示例值仅供格式参考）
    params = df.iloc[0].to_dict()

    # 情绪维度评估（基于文献的简化判断逻辑）
    assessments = []

    # 1. 压力水平评估
    stress_score = 0
    if params['HRV_SDNN'] < 50 or params['HRV_RMSSD'] < 30:
        assessments.append("当前显示出较高压力水平，自主神经系统调节能力下降")
        stress_score = 40
    else:
        assessments.append("压力水平处于正常范围，生理应激反应良好")
        stress_score = 80

    # 2. 情绪稳定性评估
    mood_score = 0
    if params['HRV_SD1'] / params['HRV_SD2'] < 0.5 or params['HRV_CSI'] > 150:
        assessments.append("情绪波动较大，可能出现焦虑或情绪调节困难")
        mood_score = 35
    else:
        assessments.append("情绪状态较为稳定，心理适应能力良好")
        mood_score = 75

    # 3. 自主神经平衡评估
    balance_score = 0
    if params['HRV_LFHF'] > 3:
        assessments.append("交感神经活动偏强，可能处于紧张或亢奋状态")
        balance_score = 45
    elif params['HRV_LFHF'] < 0.5:
        assessments.append("副交感神经活跃，需注意疲劳或能量不足状态")
        balance_score = 55
    else:
        assessments.append("自主神经系统处于良好平衡状态")
        balance_score = 85

    # 4. 心理适应能力评估
    adapt_score = 0
    if params['HRV_SampEn'] < 1.2 or params['HRV_ApEn'] < 0.8:
        assessments.append("心理生理系统灵活性不足，可能影响压力适应能力")
        adapt_score = 50
    else:
        assessments.append("表现出良好的心理生理适应性和系统复杂性")
        adapt_score = 90

    # 综合评分（加权平均）
    total_score = int((stress_score * 0.3 + mood_score * 0.25 + balance_score * 0.25 + adapt_score * 0.2))

    # 生成报告
    report = "情绪状态评估报告：\n" + "\n".join([f"{i + 1}. {item}" for i, item in enumerate(assessments)])
    report += "建议：" + get_suggestions(total_score)

    return report, total_score


def get_suggestions(score):
    if score >= 80:
        return "保持当前状态，建议维持规律作息和适度运动"
    elif score >= 60:
        return "注意压力管理，可尝试冥想或深呼吸练习"
    elif score >= 40:
        return "建议进行专业压力检测，增加休闲放松时间"
    else:
        return "建议寻求专业心理支持，及时调整身心状态"


if __name__ == '__main__':
    with open("./data/example.txt", 'r') as f:
        data = json.load(f)
        data = np.array(data)
    # ppg = nk.ppg_simulate(duration=30, sampling_rate=1000)
    _, ppg = array2ppg(data, sampling_rate=30)
    plot_ppg_signal(ppg[0])
    df = ppg_hrv(ppg[0], 30)
    for i in df.columns:
        if pd.isna(df[i].item()):
            continue
        print(i)
    print(estimate_emotions(df))
    report, score = analyze_emotion_hrv(df)
    print(report)
    print(score)
