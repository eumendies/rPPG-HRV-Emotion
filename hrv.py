import neurokit2 as nk
import pandas as pd


def ppg_hrv(ppg_signal, sampling_rate):
    """使用ppg信号计算hrv"""
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)
    ppg_peaks, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
    df_time = nk.hrv_time(ppg_peaks, sampling_rate=sampling_rate, show=False)
    df_freq = nk.hrv_frequency(ppg_peaks, sampling_rate=sampling_rate, show=False)
    return pd.concat([df_time, df_freq], axis=1)


if __name__ == '__main__':
    ppg = nk.ppg_simulate(duration=30, sampling_rate=1000)
    df = ppg_hrv(ppg, 1000)
    a = df['HRV_SDNN'].item()
    print(a)
    print(df)
