import matplotlib.pyplot as plt


def plot_ppg_signal(signal):
    """
    绘制PPG信号波形

    参数:
        signal (array-like): 形状为[2048]的PPG信号数据
    """
    plt.figure(figsize=(12, 4))  # 设置画布大小
    plt.plot(signal, color='red', linewidth=1)
    plt.grid(True, alpha=0.3)  # 网格线
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])
    plt.tight_layout()
    plt.show()
