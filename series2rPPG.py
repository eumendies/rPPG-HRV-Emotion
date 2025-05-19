"""
Author: Harryhht
Date: 2022-01-06 10:28:39
LastEditors: Eumendies
LastEditTime: 2025-04-10 14:30:00
Description:
"""
import numpy as np
import scipy
import seaborn as sns
from obspy.signal.detrend import polynomial
from sklearn.decomposition import PCA
from constants import ONE_MINUTE
from constants import FFT_HR, PEAK_HR
from scipy import signal

sns.set()


class Series2rPPG:
    """将signal转换为PPG信号，signal形状为[3, FEATURE_WINDOW, 3]"""
    def signal_preprocessing_single(self, sig):
        return polynomial(sig, order=2)

    def signal_preprocessing(self, rgbsig):
        """去趋势，相当于平滑化"""
        data = np.array(rgbsig)
        data_r = polynomial(data[:, 0], order=2)
        data_g = polynomial(data[:, 1], order=2)
        data_b = polynomial(data[:, 2], order=2)

        return np.array([data_r, data_g, data_b]).T

    def PBV_2D(self, signal):
        """
        args:
            signal: shape [T, 3]
        """
        signal = signal.T
        sig_mean = np.mean(signal, axis=1)

        sig_norm_r = signal[0, :] / sig_mean[0]
        sig_norm_g = signal[1, :] / sig_mean[1]
        sig_norm_b = signal[2, :] / sig_mean[2]

        pbv_n = np.array(
            [np.std(sig_norm_r), np.std(sig_norm_g), np.std(sig_norm_b)])
        pbv_d = np.sqrt(
            np.var(sig_norm_r) + np.var(sig_norm_g) + np.var(sig_norm_b))
        pbv = pbv_n/pbv_d

        C = np.array([sig_norm_r, sig_norm_g, sig_norm_b])
        Ct = C.T
        Q = np.matmul(C, Ct)
        W = np.linalg.solve(Q, pbv)

        A = np.matmul(Ct, W)
        B = np.matmul(pbv.T, W)
        bvp = A / B
        return bvp

    def PBV(self, signal):
        num_roi, _, _ = signal.shape
        result = []
        for i in range(num_roi):
            bvp = self.PBV_2D(signal[i])
            result.append(bvp)
        return np.array(result)


    def POS_2D(self, signal, sampling_rate, window=1.6):
        window = int(sampling_rate * window)
        H = np.full(len(signal), 0)

        for t in range(0, (signal.shape[0] - window)):
            # Spatial averaging
            C = signal[t: t + window - 1, :].T

            # Temporal normalization
            mean_color = np.mean(C, axis=1)
            try:
                Cn = np.matmul(np.linalg.inv(np.diag(mean_color)), C)
            except np.linalg.LinAlgError:  # Singular matrix
                continue

            # Projection
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)

            # Tuning (2D signal to 1D signal)
            std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
            P = np.matmul(std, S)

            # Overlap-Adding
            H[t: t + window - 1] = H[t: t + window - 1] + (P - np.mean(P)) / np.std(P)
        return H

    def POS(self, signal, sampling_rate, window=1.6):
        """参照POS_2D修改的3D版本"""
        num_roi, feature_window, channel = signal.shape
        window = int(sampling_rate * window)
        H = np.full([num_roi, feature_window], 0)

        for t in range(0, (feature_window - window)):
            # Spatial averaging
            C = signal[:, t: t + window - 1, :].transpose(0, 2, 1)  # [num_roi=3, channel=3, window]

            # Temporal normalization
            mean_color = np.mean(C, axis=-1)    # [num_roi=3, channel=3]
            try:
                diag = np.apply_along_axis(np.diag, 1, mean_color)  # 将axis=1转化为对角矩阵，最终形状[num_roi=3, channel=3, channel=3]
                Cn = np.matmul(np.linalg.inv(diag), C)  # [num_roi, channel, window]
            except np.linalg.LinAlgError:  # Singular matrix
                continue

            # Projection
            proj = np.array([[0, 1, -1], [-2, 1, 1]])
            proj = np.repeat(proj[np.newaxis, :, :], num_roi, axis=0)
            S = np.matmul(proj, Cn)     # [num_roi, 2, window]

            # Tuning
            ones_column = np.ones([num_roi, 1])
            std = (np.std(S[:, 0, :], axis=1) / np.std(S[:, 1, :], axis=1))[np.newaxis, :].transpose(1, 0)
            std = np.hstack([ones_column, std])[:, np.newaxis, :]   # [num_roi, 1, 2]
            P = np.matmul(std, S)[:, 0, :]  # [num_roi, window]

            # Overlap-Adding
            P_mean = np.mean(P, axis=-1, keepdims=True)
            P_std = np.std(P, axis=-1, keepdims=True)
            H[:, t: t + window - 1] = H[:, t: t + window - 1] + (P - P_mean) / P_std
        return H

    def CHROM(self, signal):
        X = signal
        Xcomp = 3 * X[:, :, 0] - 2 * X[:, :, 1]
        Ycomp = (1.5 * X[:, :, 0]) + X[:, :, 1] - (1.5 * X[:, :, 2])
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)
        alpha = sX / sY
        bvp = Xcomp - alpha * Ycomp
        return bvp

    def PCA(self, signal):
        bvp = []
        for i in range(signal.shape[0]):
            X = signal[i]
            pca = PCA(n_components=3)
            pca.fit(X)
            bvp.append(pca.components_[0] * pca.explained_variance_[0])
            bvp.append(pca.components_[1] * pca.explained_variance_[1])
        bvp = np.array(bvp)
        return bvp

    def GREEN(self, signal):
        """使用G通道作为最终信号"""
        return signal[:, :, 1]

    def GREEN_RED(self, signal):
        """使用G通道和R通道的线性组合作为最终信号"""
        return signal[:, :, 1] - signal[:, :, 0]

    def cal_bpm(self, pre_bpm, signal, spec, fps, mode=FFT_HR):
        if mode == FFT_HR:
            return self.calculate_fft_hr(pre_bpm, spec, fps)
        elif mode == PEAK_HR:
            return self.calculate_peak_hr(pre_bpm, signal, fps)
        else:
            return pre_bpm

    def calculate_fft_hr(self, pre_bpm, spec, fps):
        current_bpm = np.argmax(spec[:int(len(spec) / 2)]) / len(spec) * fps * ONE_MINUTE
        return pre_bpm * 0.95 + current_bpm * 0.05

    def calculate_peak_hr(self, pre_bpm, signal, fs=30):
        """Calculate heart rate based on PPG using peak detection."""
        ppg_peaks, _ = scipy.signal.find_peaks(signal)
        current_bpm = ONE_MINUTE / (np.mean(np.diff(ppg_peaks)) / fs)
        return pre_bpm * 0.95 + current_bpm * 0.05


def butterworth_filter(data, sample_rate, low=0.83, high=2.5, order=11):
    """巴特沃斯滤波器"""
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


def array2ppg(arr, sampling_rate, mode='GREEN'):
    processor = Series2rPPG()
    if mode == 'GREEN':
        bvp = processor.GREEN(arr)
    elif mode == 'GREEN-RED':
        bvp = processor.GREEN_RED(arr)
    elif mode == 'CHROM':
        bvp = processor.CHROM(arr)
    elif mode == 'PBV':
        bvp = processor.PBV(arr)
    elif mode == 'POS':
        bvp = processor.POS(arr, sampling_rate)
    else:
        bvp = processor.GREEN(arr)
    bvp_filtered = np.array([butterworth_filter(processor.signal_preprocessing_single(bvp[i, :]),
                                                     sample_rate=sampling_rate, order=5) for i in range(3)])
    return bvp, bvp_filtered


if __name__ == '__main__':
    # processor = Series2rPPG()
    # signal = np.random.rand(3, 256, 3) * 256
    # bvp = processor.PBV(signal)
    # print(bvp)
    # print(bvp.shape)

    processor = Series2rPPG()
    arr = np.random.rand(256, 3) * 256
    bvp = processor.PBV_2D(arr)
    print(bvp)
    print(bvp.shape)