import neurokit2 as nk
import numpy as np
import pandas as pd


def ppg_hrv(ppg_signal, sampling_rate):
    """使用ppg信号计算hrv"""
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)
    ppg_peaks, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
    df_time = nk.hrv_time(ppg_peaks, sampling_rate=sampling_rate, show=False)
    df_freq = nk.hrv_frequency(ppg_peaks, sampling_rate=sampling_rate, show=False)
    return pd.concat([df_time, df_freq], axis=1)


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


if __name__ == '__main__':
    ppg = nk.ppg_simulate(duration=30, sampling_rate=1000)
    df = ppg_hrv(ppg, 1000)
    a = df['HRV_SDNN'].item()
    print(estimate_emotions(df))
