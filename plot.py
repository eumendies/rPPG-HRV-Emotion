import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="SimHei.ttf", size=14)


def plot_ppg_signal(signal, show=False, output_path=None):
    """
    绘制PPG信号波形

    参数:
        signal (array-like): 形状为[3, 2048]的PPG信号数据
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    regions = ['前额', '左脸颊', '右脸颊']
    for i, (ax, region) in enumerate(zip(axes, regions)):
        ax.plot(signal[i], color='red', linewidth=1)
        ax.axis('off')
        ax.set_title(region, fontsize=14, pad=20, font=font)  # 添加区域标题

    plt.tight_layout()

    if show:
        plt.show()

    if output_path:
        file_name = output_path
        plt.savefig(
            file_name,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1
        )


