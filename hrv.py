import random

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def ppg_hrv(ppg_signal, sampling_rate):
    """使用ppg信号计算hrv"""
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)
    ppg_peaks, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
    df_time = nk.hrv_time(ppg_peaks, sampling_rate=sampling_rate, show=False)
    df_nonlinear = nk.hrv_nonlinear(ppg_peaks, sampling_rate=sampling_rate, show=False)
    df_freq = nk.hrv_frequency(ppg_peaks, sampling_rate=sampling_rate, show=False)
    return pd.concat([df_time, df_freq, df_nonlinear], axis=1)


def calculate_emotion_scores(hrv_data):
    # 定义情绪字典，包含六种情绪及其初始得分为0
    emotion_scores = {
        '愤怒': 0,
        '厌恶': 0,
        '恐惧': 0,
        '快乐': 0,
        '悲伤': 0,
        '惊讶': 0
    }

    # 获取HRV时域、频域和非线性指标
    hrv_mean_nn = hrv_data['HRV_MeanNN'][0]
    hrv_sdnn = hrv_data['HRV_SDNN'][0]
    hrv_rmssd = hrv_data['HRV_RMSSD'][0]
    hrv_sdsd = hrv_data['HRV_SDSD'][0]
    hrv_pnn50 = hrv_data['HRV_pNN50'][0]
    hrv_lf = hrv_data['HRV_LF'][0]
    hrv_hf = hrv_data['HRV_HF'][0]
    hrv_vhf = hrv_data['HRV_VHF'][0]
    hrv_tp = hrv_data['HRV_TP'][0]
    hrv_lf_hf = hrv_data['HRV_LFHF'][0]
    hrv_sd1 = hrv_data['HRV_SD1'][0]

    # 根据时域、频域和非线性指标计算情绪得分
    # 愤怒
    anger_score = 0
    if hrv_pnn50 < 0.05:  # PNN50较低与压力增加、焦虑和抑郁等心理状态有关
        anger_score += 2
    if hrv_rmssd < 50:  # RMSSD的降低可能与压力增加、焦虑和抑郁等心理状态有关
        anger_score += 3
    if hrv_lf_hf > 1.5:  # 高兴和愤怒的LF/HF值较大
        anger_score += 2
    if hrv_sdsd > np.mean(hrv_data['HRV_SDSD']):  # 负性情绪的SDSD值较大
        anger_score += 3
    anger_score = min(10, max(0, anger_score))

    # 厌恶
    disgust_score = 0
    if hrv_mean_nn < np.mean(hrv_data['HRV_MeanNN']) - 10:  # MeanNN较低可能与厌恶情绪有关
        disgust_score += 2
    if hrv_rmssd < 50:
        disgust_score += 3
    if hrv_hf < np.mean(hrv_data['HRV_HF']) - 50:  # 高频功率较低可能与厌恶情绪有关
        disgust_score += 3
    if hrv_sdsd > np.mean(hrv_data['HRV_SDSD']):
        disgust_score += 2
    disgust_score = min(10, max(0, disgust_score))

    # 恐惧
    fear_score = 0
    if hrv_mean_nn < np.mean(hrv_data['HRV_MeanNN']) - 5:  # MeanNN较低可能与恐惧情绪有关
        fear_score += 3
    if hrv_sdnn < np.mean(hrv_data['HRV_SDNN']) - 10:  # SDNN较低可能与恐惧情绪有关
        fear_score += 3
    if hrv_lf_hf < 0.5:  # 恐惧情绪下LF/HF值可能较低
        fear_score += 3
    if hrv_sd1 > np.mean(hrv_data['HRV_SD1']) + 10:  # 恐惧情绪下非线性指标可能较大
        fear_score += 2
    fear_score = min(10, max(0, fear_score))

    # 快乐
    happy_score = 0
    if hrv_pnn50 > 0.2:  # PNN50较高与积极情绪有关
        happy_score += 3
    if hrv_rmssd > 70:  # RMSSD较高与积极情绪有关
        happy_score += 3
    if hrv_lf_hf > 1.5:
        happy_score += 2
    if hrv_sdnn > np.mean(hrv_data['HRV_SDNN']) + 10:  # 高兴和悲伤的SDNN值较大
        happy_score += 3
    happy_score = min(10, max(0, happy_score))

    # 悲伤
    sad_score = 0
    if hrv_mean_nn > np.mean(hrv_data['HRV_MeanNN']) + 5:  # MeanNN较高可能与悲伤情绪有关
        sad_score += 3
    if hrv_sdnn > np.mean(hrv_data['HRV_SDNN']) + 10:
        sad_score += 3
    if hrv_tp > np.mean(hrv_data['HRV_TP']) + 50:  # 悲伤和愉悦的TP值较大
        sad_score += 3
    if hrv_hf > np.mean(hrv_data['HRV_HF']) + 50:  # 悲伤的高频功率PHF值较大
        sad_score += 2
    sad_score = min(10, max(0, sad_score))

    # 惊讶
    surprise_score = 0
    if hrv_mean_nn > np.mean(hrv_data['HRV_MeanNN']) + 5:  # MeanNN较高可能与惊讶情绪有关
        surprise_score += 2
    if hrv_rmssd > 70:
        surprise_score += 3
    if hrv_sd1 > np.mean(hrv_data['HRV_SD1']) + 10:  # 惊讶情绪下非线性指标可能较大
        surprise_score += 3
    if hrv_lf > np.mean(hrv_data['HRV_LF']) + 50:  # 惊讶情绪下低频功率可能较大
        surprise_score += 2
    surprise_score = min(10, max(0, surprise_score))

    # 添加情绪约束逻辑，快乐与悲伤、厌恶、恐惧、愤怒之间的约束
    happy_threshold = 3  # 快乐得分阈值
    if happy_score >= happy_threshold:
        # 如果快乐得分较高，降低悲伤、厌恶、恐惧、愤怒的得分
        sad_score = max(0, sad_score - 1)
        disgust_score = max(0, disgust_score - 1)
        fear_score = max(0, fear_score - 1)
        anger_score = max(0, anger_score - 1)
        # 确保其他情绪得分不超过快乐得分
        sad_score = min(sad_score, happy_score - 1)
        disgust_score = min(disgust_score, happy_score - 1)
        fear_score = min(fear_score, happy_score - 1)
        anger_score = min(anger_score, happy_score - 1)

    # 重新计算情绪分数（确保分数在0-10范围内）
    emotion_scores['愤怒'] = anger_score
    emotion_scores['厌恶'] = disgust_score
    emotion_scores['恐惧'] = fear_score
    emotion_scores['快乐'] = happy_score
    emotion_scores['悲伤'] = sad_score
    emotion_scores['惊讶'] = surprise_score

    # 检查情绪分数是否在合理范围内
    for emotion in emotion_scores:
        emotion_scores[emotion] = min(10, max(0, emotion_scores[emotion]))

    # 如果所有情绪的分数都较低，认为心情平静，将所有情绪分数设为接近0的值
    if all(score <= 2 for score in [anger_score, disgust_score, fear_score, happy_score, sad_score, surprise_score]):
        emotion_scores = {emotion: random.Random().uniform(0.1, 0.5) for emotion in emotion_scores}
    else:
        emotion_scores['愤怒'] = anger_score
        emotion_scores['厌恶'] = disgust_score
        emotion_scores['恐惧'] = fear_score
        emotion_scores['快乐'] = happy_score
        emotion_scores['悲伤'] = sad_score
        emotion_scores['惊讶'] = surprise_score

    return emotion_scores


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
    输入：包含前额、左脸颊、右脸颊三个区域HRV参数的DataFrame（三行数据）
    输出：情绪报告文本，心理评分（0-100）
    """
    # 提取三个区域数据
    forehead = df.iloc[0].to_dict()
    left_cheek = df.iloc[1].to_dict()
    right_cheek = df.iloc[2].to_dict()

    # 动态权重计算（基于信号质量指标）
    def calculate_weights(*params_list):
        # 质量评估指标：SDNN反映整体变异性，RMSSD反映高频成分，SampEn反映信号复杂度
        quality_scores = [
            p['HRV_SDNN'] * 0.4 +
            p['HRV_RMSSD'] * 0.3 +
            p['HRV_SampEn'] * 0.3
            for p in params_list
        ]
        total = sum(quality_scores)
        return [s / total if total != 0 else 1 / 3 for s in quality_scores]

    weights = calculate_weights(forehead, left_cheek, right_cheek)

    # 参数融合函数
    def integrate_params(param_name):
        vals = [forehead.get(param_name, 0),
                left_cheek.get(param_name, 0),
                right_cheek.get(param_name, 0)]
        return np.average(vals, weights=weights)

    # 综合参数计算
    integrated = {key: integrate_params(key) for key in forehead.keys()}

    # 情绪维度评估
    assessments = []

    # 1. 压力水平评估（动态阈值）
    stress_thresh = 250 + integrated['HRV_TP'] * 100  # TP越大阈值适当提高
    stress_score = 100 - (max(0, stress_thresh - integrated['HRV_SDNN']) * 1.2 +
                         max(0, (400 - integrated['HRV_RMSSD']) * 0.2))
    if integrated['HRV_SDNN'] < 50 or integrated['HRV_RMSSD'] < 30:
        assessments.append("多区域检测显示压力水平较高，建议立即放松调节")
    else:
        assessments.append("各区域压力指标均在健康范围，保持良好状态")

    # 2. 情绪稳定性评估（基于多区域一致性）
    std_ratio = np.std([forehead['HRV_SD1SD2'],
                        left_cheek['HRV_SD1SD2'],
                        right_cheek['HRV_SD1SD2']])
    mood_score = max(0, 100 - std_ratio * 50)
    if std_ratio > 1.5:
        assessments.append("多区域情绪波动差异显著，可能存在心理冲突")
    elif integrated['HRV_CSI'] > 160:
        assessments.append("综合情绪稳定性偏低，建议进行正念训练")
    else:
        assessments.append("各区域情绪信号协调，心理状态稳定")

    # 3. 神经平衡评估（动态模式分析）
    lfhf_std = np.std([forehead['HRV_LFHF'],
                       left_cheek['HRV_LFHF'],
                       right_cheek['HRV_LFHF']])
    balance_score = max(0, 100 - abs(integrated['HRV_LFHF'] - 0.5) * 20 - lfhf_std * 15)
    if lfhf_std > 0.8:
        assessments.append("自主神经调节存在区域失衡现象")
    elif integrated['HRV_LFHF'] > 2.5:
        assessments.append("交感神经活动整体偏强，注意过度紧张")
    elif integrated['HRV_LFHF'] < 0.8:
        assessments.append("副交感神经优势状态，适合恢复休整")
    else:
        assessments.append("自主神经系统处于理想平衡状态")

    # 4. 心理韧性评估（多维度复合）
    entropy_avg = (integrated['HRV_SampEn'] + integrated['HRV_ApEn']) / 2
    adapt_score = min(100, 70 + entropy_avg * 15 -
                      integrated['HRV_SI'] * 0.3)
    if adapt_score < 60:
        assessments.append("心理生理系统适应性不足，需增强应变能力")
    else:
        assessments.append("表现出优秀的心理弹性和适应能力")

    # 动态加权总分
    total_score = np.clip(
        stress_score * 0.3 +
        mood_score * 0.25 +
        balance_score * 0.25 +
        adapt_score * 0.2,
        0, 100)

    # 生成报告
    report = "情绪评估报告：\n" + "\n".join(
        [f"{i + 1}. {item}" for i, item in enumerate(assessments)])
    report += "\n"
    report += "建议：" + get_suggestions(total_score)

    return report, total_score


def get_suggestions(score):
    if score >= 85:
        return "保持当前良好状态，注意维持工作生活平衡"
    elif score >= 65:
        return "建议进行日常压力管理，每周3次有氧运动"
    elif score >= 45:
        return "推荐进行专业心理评估，配合放松训练"
    else:
        return "建议立即寻求专业心理咨询干预"


if __name__ == '__main__':
    # with open("./data/example.txt", 'r') as f:
    #     data = json.load(f)
    #     data = np.array(data)
    # # ppg = nk.ppg_simulate(duration=3000, sampling_rate=1000)
    # # df = ppg_hrv(ppg, 1000)
    # # df = pd.concat([df, df, df], axis=0)
    # _, ppg = array2ppg(data, sampling_rate=30)
    # plot_ppg_signal(ppg)
    # df = pd.concat([pd.DataFrame(ppg_hrv(ppg[i], 30)) for i in range(3)], axis=0)
    # for i in df.columns:
    #     if pd.isna(df[i].iloc[0]):
    #         continue
    #     print(i)
    # print(estimate_emotions(df))
    # report, score = analyze_emotion_hrv(df)
    # print(report)
    # print(score)

    ppg = nk.ppg_simulate(duration=30, sampling_rate=1000)
    # 调用函数计算情绪得分
    emotion_scores = calculate_emotion_scores(ppg_hrv(ppg, 1000))
    print(emotion_scores)
