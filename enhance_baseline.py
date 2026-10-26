# -*- coding: utf-8 -*-
"""
数字图像处理大作业 —— 基础版（Baseline）
==========================================
任务：低光照 / 低对比度彩色图像的增强

数据准备（make_low_light）：
    在 HSV 空间只对亮度通道 V 做伽马(γ=2.5)变暗，
    色调 H、饱和度 S 保持不变，以此模拟低光照采集图像。

基础版增强（enhance_baseline）：
    直接对彩色图像的 B、G、R 三个通道分别做
    【全局直方图均衡化（Global Histogram Equalization, HE）】。

运行：python enhance_baseline.py
依赖：opencv-python、numpy
"""
import os
import cv2
import numpy as np


def _gamma_table(gamma):
    """生成 0~255 的伽马变换查找表(LUT)。"""
    return np.array([((i / 255.0) ** gamma) * 255.0
                     for i in np.arange(256)]).astype("uint8")


def make_low_light(img_bgr, gamma=2.5):
    """
    在 HSV 空间模拟低光照：只把亮度通道 V 变暗，H、S 保持不变。
    公式： V_dark = 255 * (V / 255) ^ gamma （gamma > 1 变暗）。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_dark = cv2.LUT(v, _gamma_table(gamma))
    return cv2.cvtColor(cv2.merge((h, s, v_dark)), cv2.COLOR_HSV2BGR)


def enhance_baseline(img_bgr):
    """
    基础版增强：把 BGR 三通道拆开，对每个通道【独立】做全局直方图均衡化，
    再合并回彩色图像。
    直方图均衡化：以累积分布函数(CDF)作为映射，把灰度分布拉伸到整个 0~255。
    """
    b, g, r = cv2.split(img_bgr)          # 拆分为三个单通道
    b_eq = cv2.equalizeHist(b)            # 对 B 通道做全局均衡化
    g_eq = cv2.equalizeHist(g)            # 对 G 通道做全局均衡化
    r_eq = cv2.equalizeHist(r)            # 对 R 通道做全局均衡化
    return cv2.merge((b_eq, g_eq, r_eq))  # 合并为增强后的彩色图像


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(os.path.join(here, "data", "lena.png"))
    low = make_low_light(img)
    enhanced = enhance_baseline(low)
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "baseline_lena_lowlight.png"), low)
    cv2.imwrite(os.path.join(out_dir, "baseline_lena_enhanced.png"), enhanced)
    print("[Baseline] 已保存基础版结果到 results/ 目录")


if __name__ == "__main__":
    main()
