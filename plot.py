import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="SimHei.ttf", size=14)


def plot_ppg_signal(signal):
    """
    绘制PPG信号波形

    参数:
        signal (array-like): 形状为[3, 2048]的PPG信号数据
    """
    regions = ['前额', '左脸颊', '右脸颊']
    for i, region in enumerate(regions):
        ax = plt.subplot2grid((3, 2), (i, 1), colspan=1)
        ax.plot(signal[i], color='red', linewidth=1)
        ax.axis('off')
        ax.set_title(region, fontsize=14, pad=20, font=font)  # 添加区域标题


def plot_emotion_radar(emojis_scores):
    """
    绘制情绪六边形评分图

    参数:
    emojis_scores: 字典，包含六个情绪的得分，格式如 {'愤怒': 7, '悲伤': 5, '快乐': 9...}
    """
    # 确保有六个情绪
    if len(emojis_scores) != 6:
        raise ValueError("字典必须包含六个情绪")

    # 计算六边形的theta值（角度）
    num_vars = len(emojis_scores)
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    # 将字典转换为列表
    emotions = list(emojis_scores.keys())
    scores = list(emojis_scores.values())

    # 重复一次第一个值，使图形闭合
    scores += scores[:1]
    theta = theta.tolist()
    theta += theta[:1]

    # 创建极坐标图
    ax = plt.subplot2grid((1, 2), (0, 0), colspan=1, polar=True)

    # 绘制六边形
    ax.fill(theta, scores, color='blue', alpha=0.25)
    ax.plot(theta, scores, color='blue')

    # 设置最大值和最小值
    ax.set_ylim(0, 10)

    # 设置标签
    ax.set_yticks(range(0, 11, 2))
    ax.set_yticklabels(range(0, 11, 2), color='grey', size=12)

    labels_with_scores = [f"{emotion}\n{score}" for emotion, score in zip(emotions, scores[:-1])]
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels_with_scores, color='black', size=6, font=font)

    ax.set_title('情绪评分六边形图', size=20, color='black', y=1.1, font=font)


def plot_emotion_and_ppg(emotion_scores, signal, show=False, output_path=None):
    plt.figure(figsize=(15, 10))

    plot_emotion_radar(emotion_scores)
    plot_ppg_signal(signal)
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
