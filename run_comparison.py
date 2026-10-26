# -*- coding: utf-8 -*-
"""
数字图像处理大作业 —— 结果对比与评估
======================================
对每张标准测试图像：
  原图(参考) -> 低光照(输入) -> 基础版增强 -> 改进版增强
输出：
  1) 每幅图的 4 联对比图；
  2) 灰度直方图对比图（以 Lena 为例）；
  3) 定量指标：平均亮度、标准差(对比度)、信息熵、平均梯度(清晰度)。

运行：python run_comparison.py
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enhance_baseline import make_low_light, enhance_baseline
from enhance_improved import enhance_improved

# Windows 下让 matplotlib 正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMAGES = ["lena.png", "baboon.jpg", "fruits.jpg"]
TITLES = ["原图（正常光照参考）", "低光照图像（增强输入）",
          "基础版：RGB三通道全局HE", "改进版：HSV-V通道CLAHE+伽马提亮"]


def calc_metrics(img_bgr):
    """返回 (平均亮度, 标准差, 信息熵, 平均梯度)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean = gray.mean()                       # 平均亮度
    std = gray.std()                         # 标准差 -> 对比度
    # 信息熵
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -(p * np.log2(p)).sum()
    # 平均梯度 -> 清晰度
    gx = np.diff(gray, axis=1)[:-1, :]
    gy = np.diff(gray, axis=0)[:, :-1]
    grad = np.sqrt(gx ** 2 + gy ** 2).mean()
    return mean, std, entropy, grad


def save_comparison(here, name, panels):
    """保存 1x4 联对比图"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, img, title in zip(axes, panels, TITLES):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(here, "results", f"comparison_{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def save_histogram(here, low, base, imp):
    """保存灰度直方图对比（低光照 / 基础版 / 改进版）"""
    def hist(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        return h / h.sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist(low), color="gray", label="低光照输入")
    ax.plot(hist(base), color="red", label="基础版 RGB-HE")
    ax.plot(hist(imp), color="blue", label="改进版 HSV-CLAHE")
    ax.set_xlabel("灰度级"); ax.set_ylabel("归一化概率")
    ax.set_title("Lena 灰度直方图对比")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(here, "results", "histogram_lena.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows = {}
    for fname in IMAGES:
        name = os.path.splitext(fname)[0]
        orig = cv2.imread(os.path.join(here, "data", fname))
        low = make_low_light(orig)
        base = enhance_baseline(low)
        imp = enhance_improved(low)

        # 保存单幅结果，便于报告引用
        cv2.imwrite(os.path.join(here, "results", f"{name}_lowlight.png"), low)
        cv2.imwrite(os.path.join(here, "results", f"{name}_baseline.png"), base)
        cv2.imwrite(os.path.join(here, "results", f"{name}_improved.png"), imp)

        save_comparison(here, name, [orig, low, base, imp])
        rows[name] = {
            "低光照": calc_metrics(low),
            "基础版": calc_metrics(base),
            "改进版": calc_metrics(imp),
        }
        if name == "lena":
            save_histogram(here, low, base, imp)

    # 打印指标表（供报告引用）
    header = f"{'图像':<8}{'版本':<8}{'平均亮度':>10}{'标准差':>10}{'信息熵':>10}{'平均梯度':>10}"
    lines = [header]
    for name, d in rows.items():
        for ver, m in d.items():
            lines.append(f"{name:<8}{ver:<8}{m[0]:>10.2f}{m[1]:>10.2f}{m[2]:>10.3f}{m[3]:>10.2f}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(here, "results", "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(table)
    print("\n[OK] 对比图、直方图、指标已保存到 results/ 目录")


if __name__ == "__main__":
    main()
